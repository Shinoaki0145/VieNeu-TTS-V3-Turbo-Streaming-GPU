import argparse
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
import threading
import time
import webbrowser
import traceback
import uuid
from datetime import datetime
from datetime import timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vieneu import Vieneu
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

SAMPLE_RATE = 48_000
FRAME_JSON = 1
FRAME_AUDIO = 2
MAX_BODY_BYTES = 65_536
MAX_TEXT_LENGTH = 20_000
SAMPLING_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 25,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_chars": 256,
    "max_new_frames": 300,
}
SAMPLING_LIMITS = {
    "temperature": {"min": 0.1, "max": 1.5, "step": 0.05},
    "top_k": {"min": 1, "max": 100, "step": 1},
    "top_p": {"min": 0.05, "max": 1.0, "step": 0.05},
    "repetition_penalty": {"min": 1.0, "max": 2.0, "step": 0.05},
    "max_chars": {"min": 128, "max": 512, "step": 1},
    "max_new_frames": {"min": 1, "max": 1200, "step": 1},
}
REQUEST_FIELDS = {"text", "voice", *SAMPLING_DEFAULTS}
UI_PATH = Path(__file__).with_name("stream_ui.html")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VieNeu-TTS v3 Turbo streaming qua trình duyệt Windows"
    )
    parser.add_argument(
        "--text",
        default=(
            "Xin chào các bạn. Đây là bài kiểm tra streaming thời gian thực "
            "của VieNeu TTS phiên bản ba Turbo."
        ),
        help="Văn bản cần đọc",
    )
    parser.add_argument("--voice", default="Mai Anh", help="Tên preset voice")
    parser.add_argument(
        "--save",
        default=None,
        help="Nếu đặt, lưu toàn bộ audio ra WAV, ví dụ output_stream.wav",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm-up GPU một lần trước khi phát",
    )
    parser.add_argument("--port", type=int, default=8001, help="Cổng web local")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Không tự mở trình duyệt",
    )
    return parser.parse_args()


def normalize_voices(records):
    voices = []
    for record in records:
        if isinstance(record, (tuple, list)) and len(record) >= 2:
            label, value = record[0], record[1]
        else:
            label = value = record
        voices.append({"label": str(label), "value": str(value)})
    return voices


def build_config(tts, args, hardware=None):
    return {
        "sample_rate": SAMPLE_RATE,
        "voices": normalize_voices(tts.list_preset_voices()),
        "defaults": {
            "text": args.text,
            "voice": args.voice,
            **SAMPLING_DEFAULTS,
        },
        "limits": {"text_max_length": MAX_TEXT_LENGTH, **SAMPLING_LIMITS},
        "hardware": hardware if hardware is not None else hardware_info(),
    }


def validate_stream_request(value, config):
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    unknown = set(value) - REQUEST_FIELDS
    if unknown:
        raise ValueError(f"unknown field: {sorted(unknown)[0]}")
    missing = REQUEST_FIELDS - set(value)
    if missing:
        raise ValueError(f"missing field: {sorted(missing)[0]}")

    text = value["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("text must contain at most 20,000 characters")

    voice = value["voice"]
    allowed_voices = {item["value"] for item in config["voices"]}
    if not isinstance(voice, str) or voice not in allowed_voices:
        raise ValueError("voice is not a known preset")

    result = {"text": text, "voice": voice}
    for name, bounds in SAMPLING_LIMITS.items():
        raw = value[name]
        integer = name in {"top_k", "max_chars", "max_new_frames"}
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if isinstance(raw, float) and not math.isfinite(raw):
            raise ValueError(f"{name} has an invalid numeric value")
        if integer and isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{name} has an invalid numeric value")
        if not bounds["min"] <= raw <= bounds["max"]:
            raise ValueError(
                f"{name} must be between {bounds['min']} and {bounds['max']}"
            )
        number = int(raw) if integer else float(raw)
        result[name] = number
    return result


def text_chunk_records(text, max_chars):
    chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
    return [
        {"index": index, "characters": len(chunk), "text": chunk}
        for index, chunk in enumerate(chunks, 1)
    ]


def json_payload(event):
    return json.dumps(
        event, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def encode_frame(frame_type, payload):
    return struct.pack("<BI", frame_type, len(payload)) + payload


def print_result(event):
    print("\n===== RESULT =====")
    print(f"Audio duration : {event['audio_duration_s']:.2f} s")
    print(f"Generation time: {event['generation_time_s']:.2f} s")
    rtf = event["rtf"]
    speed = event["realtime_speed"]
    first_chunk = event["first_chunk_ms"]
    print(f"RTF            : {rtf:.3f}" if rtf is not None else "RTF            : n/a")
    print(
        f"Realtime speed : {speed:.2f}x"
        if speed is not None else "Realtime speed : n/a"
    )
    print(
        f"First chunk    : {first_chunk:.1f} ms"
        if first_chunk is not None else "First chunk    : n/a"
    )


def iter_stream_frames(
    tts, params, *, synchronize, perf_counter=time.perf_counter,
    wall_now=lambda: datetime.now().astimezone(), save_path=None, save_lock=None,
):
    request_id = uuid.uuid4().hex
    run_perf = perf_counter()
    run_wall = wall_now()
    iterator = None
    try:
        records = text_chunk_records(params["text"], params["max_chars"])
        yield FRAME_JSON, json_payload({
            "event": "run_started", "request_id": request_id,
            "started_at": run_wall.isoformat(timespec="milliseconds"),
            "config": dict(params), "text_chunks": records,
        })
        iterator = iter(tts.infer_stream(
            params["text"], voice=params["voice"],
            temperature=params["temperature"], top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            max_chars=params["max_chars"],
            max_new_frames=params["max_new_frames"],
        ))
        audio_parts = []
        total_samples = 0
        first_chunk_ms = None
        chunk_index = 0

        while True:
            chunk_start_perf = perf_counter()
            chunk_start_wall = wall_now()
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            chunk_end_perf = perf_counter()
            chunk_end_wall = wall_now()
            audio = np.clip(
                np.asarray(chunk, dtype="<f4").reshape(-1), -1.0, 1.0
            )
            chunk_index += 1
            total_samples += audio.size
            if save_path:
                audio_parts.append(audio.copy())
            end_offset_ms = (chunk_end_perf - run_perf) * 1000
            if first_chunk_ms is None:
                first_chunk_ms = end_offset_ms
            yield FRAME_JSON, json_payload({
                "event": "audio_chunk", "index": chunk_index,
                "samples": int(audio.size),
                "audio_ms": audio.size / SAMPLE_RATE * 1000,
                "started_at": chunk_start_wall.isoformat(timespec="milliseconds"),
                "finished_at": chunk_end_wall.isoformat(timespec="milliseconds"),
                "start_offset_ms": (chunk_start_perf - run_perf) * 1000,
                "end_offset_ms": end_offset_ms,
                "generation_ms": (chunk_end_perf - chunk_start_perf) * 1000,
            })
            yield FRAME_AUDIO, audio.tobytes()

        synchronize()
        finished_perf = perf_counter()
        finished_wall = wall_now()
        generation_time = finished_perf - run_perf
        audio_duration = total_samples / SAMPLE_RATE
        rtf = generation_time / audio_duration if audio_duration else None
        speed = (
            audio_duration / generation_time
            if generation_time and audio_duration else None
        )
        if save_path and audio_parts:
            lock = save_lock or threading.Lock()
            with lock:
                sf.write(save_path, np.concatenate(audio_parts), SAMPLE_RATE)

        completed = {
            "event": "run_completed", "request_id": request_id,
            "finished_at": finished_wall.isoformat(timespec="milliseconds"),
            "first_chunk_ms": first_chunk_ms,
            "audio_duration_s": audio_duration,
            "generation_time_s": generation_time,
            "rtf": rtf, "realtime_speed": speed,
            "text_chunk_count": len(records),
            "audio_chunk_count": chunk_index,
            "total_samples": total_samples,
        }
        print_result(completed)
        yield FRAME_JSON, json_payload(completed)
    except GeneratorExit:
        raise
    except Exception as error:
        traceback.print_exc()
        yield FRAME_JSON, json_payload({"event": "error", "message": str(error)})
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close:
                close()


def _detected(call):
    try:
        return call()
    except Exception:
        return None


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or None


def _ram_total_bytes():
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def hardware_info():
    kernel = _detected(platform.release)
    cuda_available = bool(_detected(torch.cuda.is_available))
    gpu = None
    if cuda_available:
        capability = _detected(lambda: torch.cuda.get_device_capability(0))
        memory = _detected(lambda: torch.cuda.mem_get_info(0))
        gpu = {
            "name": _detected(lambda: torch.cuda.get_device_name(0)),
            "compute_capability": (
                f"{capability[0]}.{capability[1]}" if capability else None
            ),
            "vram_total_bytes": int(memory[1]) if memory else None,
            "vram_free_bytes": int(memory[0]) if memory else None,
        }
    return {
        "os": _detected(platform.system),
        "kernel": kernel,
        "wsl": "microsoft" in (kernel or "").lower(),
        "cpu": _detected(_cpu_name),
        "ram_total_bytes": _detected(_ram_total_bytes),
        "python": _detected(platform.python_version),
        "vieneu": _detected(lambda: importlib.metadata.version("vieneu")),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
    }


def create_server(
    tts, args, host="127.0.0.1", port=8001, synchronize=None,
    perf_counter=None, wall_now=None, hardware=None, ui_path=UI_PATH,
):
    if host != "127.0.0.1":
        raise ValueError("server host must be 127.0.0.1")
    ui_bytes = Path(ui_path).resolve().read_bytes()
    config = build_config(tts, args, hardware=hardware)
    # ponytail: one lock serializes WAV writes; use per-path locks if throughput matters.
    save_lock = threading.Lock()
    synchronize = synchronize or torch.cuda.synchronize
    perf_counter = perf_counter or time.perf_counter
    wall_now = wall_now or (lambda: datetime.now().astimezone())

    def send_json(handler, status, value):
        payload = json_payload(value)
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(payload)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(ui_bytes)))
                self.end_headers()
                self.wfile.write(ui_bytes)
                return

            if self.path == "/api/config":
                send_json(self, 200, config)
                return

            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/stream":
                self.send_error(404)
                return

            media_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if media_type.strip().lower() != "application/json":
                send_json(self, 415, {"error": "Content-Type must be application/json"})
                return

            try:
                length_header = self.headers["Content-Length"]
                if (
                    not isinstance(length_header, str)
                    or not length_header.isascii()
                    or not length_header.isdecimal()
                ):
                    raise ValueError
                length = int(length_header)
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                send_json(self, 400, {"error": "invalid Content-Length"})
                return

            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                request = json.loads(body.decode("utf-8"))
                params = validate_stream_request(request, config)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
                send_json(self, 400, {"error": str(error)})
                return

            frames = None
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-vieneu-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                frames = iter_stream_frames(
                    tts, params, synchronize=synchronize, perf_counter=perf_counter,
                    wall_now=wall_now, save_path=args.save, save_lock=save_lock,
                )
                for frame_type, payload in frames:
                    self.wfile.write(encode_frame(frame_type, payload))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("Trình duyệt đã ngắt stream.")
            finally:
                if frames is not None:
                    try:
                        frames.close()
                    except Exception:
                        pass

        def log_message(self, _format, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def open_browser(url):
    powershell = Path(
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    if powershell.exists():
        subprocess.Popen(
            [str(powershell), "-NoProfile", "-Command", "Start-Process", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        webbrowser.open(url)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng. VieNeu đang không thấy GPU NVIDIA.")

    print("===== ENV =====")
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Voice:", args.voice)
    print()

    print("Loading VieNeu v3 Turbo...")
    tts = Vieneu()

    if args.warmup:
        print("Warm-up GPU...")
        for _ in tts.infer_stream("Đây là câu khởi động GPU.", voice=args.voice):
            pass
        torch.cuda.synchronize()
        print("Warm-up complete.\n")

    try:
        server = create_server(tts, args, port=args.port, ui_path=UI_PATH)
    except FileNotFoundError:
        print(f"Không tìm thấy giao diện: {UI_PATH}", file=sys.stderr)
        raise
    except OSError as error:
        print(
            f"Không thể dùng cổng {args.port} (có thể đang được dùng): {error}",
            file=sys.stderr,
        )
        raise
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        f"Mở {url} rồi cấu hình và bấm 'Phát stream'. "
        "Nhấn Ctrl+C để dừng server."
    )
    if not args.no_browser:
        open_browser(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
