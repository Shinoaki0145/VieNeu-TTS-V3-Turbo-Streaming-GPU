import contextlib
import http.client
import io
import json
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

import stream_play


class DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags_by_id = {}
        self.label_targets = set()
        self.attrs_by_id = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.tags_by_id[attrs["id"]] = tag
            self.attrs_by_id[attrs["id"]] = attrs
        if tag == "label" and "for" in attrs:
            self.label_targets.add(attrs["for"])


class DashboardFileTest(unittest.TestCase):
    def test_dashboard_has_controls_metrics_tables_and_live_status(self):
        parser = DashboardParser()
        source = Path("stream_ui.html").read_text(encoding="utf-8")
        parser.feed(source)
        controls = {
            "text", "voice", "temperature", "top_k", "top_p",
            "repetition_penalty", "max_chars", "max_new_frames",
        }
        self.assertTrue(controls <= parser.label_targets)
        self.assertEqual(parser.tags_by_id["text"], "textarea")
        self.assertEqual(parser.tags_by_id["voice"], "select")
        for control in controls - {"text", "voice"}:
            self.assertEqual(parser.tags_by_id[control], "input")
        for button in ("play", "stop", "reset"):
            self.assertEqual(parser.tags_by_id[button], "button")
        for table_body in ("text-chunks-body", "audio-chunks-body"):
            self.assertEqual(parser.tags_by_id[table_body], "tbody")
        for metric in (
            "metric-ttfa", "metric-duration", "metric-generation",
            "metric-rtf", "metric-speed", "metric-text-chunks",
            "metric-audio-chunks",
        ):
            self.assertIn(metric, parser.tags_by_id)
        self.assertEqual(parser.tags_by_id["hardware"], "dl")
        self.assertEqual(parser.attrs_by_id["status"]["role"], "status")
        self.assertEqual(parser.attrs_by_id["status"]["aria-live"], "polite")
        self.assertIn('<label for="max_chars">Max chars</label>', source)

    def test_dashboard_preserves_exact_valid_numeric_input_values(self):
        source = Path("stream_ui.html").read_text(encoding="utf-8")

        self.assertNotIn("parseInt(", source)
        self.assertIn("const value = $(id).valueAsNumber;", source)
        self.assertIn("Number.isFinite(value)", source)
        self.assertIn("Number.isInteger(value)", source)
        self.assertIn("payload[id] = value;", source)

    def test_dashboard_updates_interim_metrics_for_each_audio_chunk(self):
        source = Path("stream_ui.html").read_text(encoding="utf-8")

        self.assertIn("interimMetrics", source)
        self.assertRegex(
            source,
            r'(?s)case "audio_chunk":.*?updateInterimMetrics\(event\);',
            msg="audio chunks must update the live summary metrics",
        )
        self.assertGreaterEqual(
            source.count("resetInterimMetrics()"),
            3,
            "interim metric state must be reset between runs",
        )
        for formula in (
            "interimMetrics.audioMs += audioMs;",
            "interimMetrics.generationMs = generationMs;",
            'format(interimMetrics.firstChunkMs, 1, " ms")',
            'format(interimMetrics.audioMs / 1000, 2, " s")',
            'format(interimMetrics.generationMs / 1000, 2, " s")',
            "interimMetrics.generationMs / interimMetrics.audioMs",
            "interimMetrics.audioMs / interimMetrics.generationMs",
        ):
            self.assertIn(formula, source)

    def test_dashboard_runtime_behaviors_when_node_is_available(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed; static dashboard contracts remain active")

        source = Path("stream_ui.html").read_text(encoding="utf-8")
        script = source.split('<script type="module">', 1)[1].split("</script>", 1)[0]
        harness = f"""
const vm = require("vm");
const {{TextDecoder}} = require("util");
const values = {{
  text: "Xin chào", voice: "Mai Anh", temperature: "0.8", top_k: "1e2",
  top_p: "0.95", repetition_penalty: "1.2", max_chars: "2e2",
  max_new_frames: "1200"
}};
const element = (value = "") => ({{
  value, textContent: "", disabled: false,
  get valueAsNumber() {{ return Number(this.value); }},
  reportValidity() {{ return true; }}, addEventListener() {{}}, append() {{}},
  appendChild() {{}}, replaceChildren() {{}}, setAttribute() {{}}
}});
const elements = Object.fromEntries(Object.entries(values).map(([id, value]) => [id, element(value)]));
for (const id of ["play", "stop", "reset", "metric-ttfa", "metric-duration", "metric-generation", "metric-rtf", "metric-speed", "metric-text-chunks", "metric-audio-chunks", "text-chunks-body", "audio-chunks-body", "hardware"]) elements[id] = element();
const sandbox = {{
  TextDecoder, console, setTimeout, clearTimeout,
  window: {{addEventListener() {{}}}},
  document: {{
    getElementById: id => elements[id], addEventListener() {{}},
    querySelectorAll: () => [], createElement: () => element()
  }}
}};
vm.runInNewContext({json.dumps(script)} + "\\nglobalThis.dashboard = {{requestPayload, resetInterimMetrics, updateInterimMetrics, renderEvent}};", sandbox);
const assert = require("assert");
assert.deepStrictEqual(JSON.parse(JSON.stringify(sandbox.dashboard.requestPayload())), {{
  text: "Xin chào", voice: "Mai Anh", temperature: 0.8, top_k: 100,
  top_p: 0.95, repetition_penalty: 1.2, max_chars: 200,
  max_new_frames: 1200
}});
sandbox.dashboard.resetInterimMetrics();
sandbox.dashboard.updateInterimMetrics({{audio_ms: 500, end_offset_ms: 1250}});
assert.deepStrictEqual(
  ["metric-ttfa", "metric-duration", "metric-generation", "metric-rtf", "metric-speed"].map(id => elements[id].textContent),
  ["1250.0 ms", "0.50 s", "1.25 s", "2.500", "0.40x"]
);
sandbox.dashboard.renderEvent({{event: "run_completed", first_chunk_ms: 1500, audio_duration_s: 2, generation_time_s: 2.5, rtf: 1.25, realtime_speed: 0.8, text_chunk_count: 1, audio_chunk_count: 4}});
assert.deepStrictEqual(
  ["metric-ttfa", "metric-duration", "metric-generation", "metric-rtf", "metric-speed"].map(id => elements[id].textContent),
  ["1500.0 ms", "2.00 s", "2.50 s", "1.250", "0.80x"]
);
sandbox.dashboard.resetInterimMetrics();
sandbox.dashboard.updateInterimMetrics({{audio_ms: 100, end_offset_ms: 200}});
assert.strictEqual(elements["metric-ttfa"].textContent, "200.0 ms");
"""
        result = subprocess.run(
            [node, "-e", harness], text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class FakeTTS:
    def __init__(self, chunks=None, voices=None, error=None):
        self.chunks = chunks if chunks is not None else [
            np.array([0.25, 1.5], dtype=np.float32),
            np.array([-2.0, 0.5], dtype=np.float32),
        ]
        self.voices = voices if voices is not None else [
            ("Mai Anh", "Mai Anh"), "Minh Quân"
        ]
        self.error = error
        self.calls = []

    def list_preset_voices(self):
        return self.voices

    def infer_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class CloseAwareIterator:
    def __init__(self, chunks, error=None):
        self._chunks = iter(chunks)
        self.error = error
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if self.error:
                raise self.error
            raise

    def close(self):
        self.closed = True


class CloseAwareTTS(FakeTTS):
    def __init__(self, iterator):
        super().__init__()
        self.iterator = iterator

    def infer_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.iterator


class InferConstructionFailureTTS(FakeTTS):
    def infer_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        raise RuntimeError("infer setup failed")


class ConfigTest(unittest.TestCase):
    def test_normalize_voices_accepts_pairs_and_scalars(self):
        self.assertEqual(
            stream_play.normalize_voices([("Mai Anh", "mai-anh"), "Minh Quân"]),
            [
                {"label": "Mai Anh", "value": "mai-anh"},
                {"label": "Minh Quân", "value": "Minh Quân"},
            ],
        )

    def test_build_config_uses_cli_defaults_and_exact_limits(self):
        args = SimpleNamespace(
            text="Văn bản mặc định", voice="Mai Anh", save=None,
            warmup=False, port=8001, no_browser=True,
        )
        hardware = {
            "os": "Linux", "kernel": "WSL2", "wsl": True,
            "cpu": "Test CPU", "ram_total_bytes": 8_000_000_000,
            "python": "3.12.14", "vieneu": "3.8.0",
            "torch": "2.8.0+cu128", "cuda_runtime": "12.8",
            "gpu": {"name": "Test GPU", "compute_capability": "12.0",
                    "vram_total_bytes": 8_000_000_000,
                    "vram_free_bytes": 7_000_000_000},
        }
        result = stream_play.build_config(FakeTTS(), args, hardware=hardware)
        self.assertEqual(result["sample_rate"], 48_000)
        self.assertEqual(result["voices"][0], {"label": "Mai Anh", "value": "Mai Anh"})
        self.assertEqual(result["defaults"], {
            "text": "Văn bản mặc định", "voice": "Mai Anh",
            "temperature": 0.8, "top_k": 25, "top_p": 0.95,
            "repetition_penalty": 1.2, "max_chars": 256,
            "max_new_frames": 300,
        })
        self.assertEqual(result["limits"]["temperature"],
                         {"min": 0.1, "max": 1.5, "step": 0.05})
        self.assertEqual(result["limits"]["text_max_length"], 20_000)
        self.assertEqual(result["limits"]["max_chars"],
                         {"min": 128, "max": 512, "step": 1})
        self.assertEqual(result["limits"]["max_new_frames"],
                         {"min": 1, "max": 1200, "step": 1})
        self.assertEqual(result["hardware"], hardware)
        self.assertEqual(set(result["limits"]), {
            "text_max_length",
            "temperature", "top_k", "top_p", "repetition_penalty", "max_chars",
            "max_new_frames",
        })
        json.dumps(result, allow_nan=False)


def valid_request():
    return {
        "text": "Xin chào", "voice": "Mai Anh", "temperature": 0.8,
        "top_k": 25, "top_p": 0.95,
        "repetition_penalty": 1.2, "max_chars": 256,
        "max_new_frames": 300,
    }


def decode_frames(data):
    frames = []
    offset = 0
    while offset < len(data):
        frame_type, size = struct.unpack_from("<BI", data, offset)
        offset += 5
        payload = data[offset:offset + size]
        if len(payload) != size:
            raise AssertionError("truncated test frame")
        frames.append((frame_type, payload))
        offset += size
    return frames


class FramingTest(unittest.TestCase):
    def test_encode_frame_uses_five_byte_little_endian_header(self):
        self.assertEqual(
            stream_play.encode_frame(2, b"abcd"),
            b"\x02\x04\x00\x00\x00abcd",
        )

    def test_json_payload_is_compact_utf8_and_rejects_nan(self):
        self.assertEqual(
            stream_play.json_payload({"event": "x", "value": "Tiếng Việt"}),
            '{"event":"x","value":"Tiếng Việt"}'.encode(),
        )
        with self.assertRaises(ValueError):
            stream_play.json_payload({"value": float("nan")})

    def test_frames_can_be_reassembled_when_every_network_chunk_is_one_byte(self):
        wire = (
            stream_play.encode_frame(1, b'{"event":"x"}')
            + stream_play.encode_frame(2, b"\x00\x00\x00\x00")
        )
        pending = bytearray()
        decoded = []
        for byte in wire:
            pending.append(byte)
            while len(pending) >= 5:
                frame_type, size = struct.unpack_from("<BI", pending)
                if len(pending) < 5 + size:
                    break
                decoded.append((frame_type, bytes(pending[5:5 + size])))
                del pending[:5 + size]
        self.assertEqual(pending, b"")
        self.assertEqual(decoded, [(1, b'{"event":"x"}'),
                                   (2, b"\x00\x00\x00\x00")])


class Sequence:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def json_events(frames):
    return [json.loads(payload) for kind, payload in frames
            if kind == stream_play.FRAME_JSON]


class StreamFramesTest(unittest.TestCase):
    def test_success_emits_ordered_telemetry_and_clipped_audio(self):
        tts = FakeTTS(chunks=[
            np.array([0.25, 1.5], dtype=np.float32),
            np.array([-2.0, 0.5], dtype=np.float32),
        ])
        perf = Sequence([0.0, 0.001, 0.111, 0.112, 0.222, 0.223, 0.300])
        wall = Sequence([
            datetime(2026, 9, 16, 14, 30, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 1000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 111000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 112000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 222000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 223000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 300000, tzinfo=timezone.utc),
        ])
        synchronize_calls = []
        params = valid_request()
        params["max_new_frames"] = 1200
        frames = list(stream_play.iter_stream_frames(
            tts, params, synchronize=lambda: synchronize_calls.append(True),
            perf_counter=perf, wall_now=wall,
        ))

        self.assertEqual([kind for kind, _ in frames], [1, 1, 2, 1, 2, 1])
        events = json_events(frames)
        self.assertEqual([e["event"] for e in events],
                         ["run_started", "audio_chunk", "audio_chunk", "run_completed"])
        self.assertEqual(events[1]["samples"], 2)
        self.assertAlmostEqual(events[1]["generation_ms"], 110.0)
        self.assertAlmostEqual(events[1]["end_offset_ms"], 111.0)
        self.assertAlmostEqual(events[2]["generation_ms"], 110.0)
        self.assertAlmostEqual(events[-1]["first_chunk_ms"], 111.0)
        self.assertEqual(events[-1]["total_samples"], 4)
        self.assertAlmostEqual(events[-1]["audio_duration_s"], 4 / 48_000)
        self.assertAlmostEqual(events[-1]["generation_time_s"], 0.300)
        self.assertAlmostEqual(events[-1]["rtf"], 0.300 / (4 / 48_000))
        self.assertAlmostEqual(events[-1]["realtime_speed"], (4 / 48_000) / 0.300)
        audio = [np.frombuffer(payload, dtype="<f4") for kind, payload in frames if kind == 2]
        np.testing.assert_array_equal(audio[0], [0.25, 1.0])
        np.testing.assert_array_equal(audio[1], [-1.0, 0.5])
        self.assertEqual(synchronize_calls, [True])
        self.assertEqual(tts.calls[0][1], {
            "voice": "Mai Anh", "temperature": 0.8, "top_k": 25,
            "top_p": 0.95, "repetition_penalty": 1.2, "max_chars": 256,
            "max_new_frames": 1200,
        })


class StreamRobustnessTest(unittest.TestCase):
    def clocks(self):
        return (
            Sequence([0.0, 0.001, 0.011, 0.012, 0.020]),
            Sequence([
                datetime(2026, 9, 16, 14, 30, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 1000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 11000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 12000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 20000, tzinfo=timezone.utc),
            ]),
        )

    def test_generator_failure_emits_error_and_closes_inner_iterator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            inner = CloseAwareIterator(
                [np.array([0.25], dtype=np.float32)], RuntimeError("codec failed")
            )
            perf, wall = self.clocks()
            with contextlib.redirect_stderr(io.StringIO()):
                frames = list(stream_play.iter_stream_frames(
                    CloseAwareTTS(inner), valid_request(), synchronize=lambda: None,
                    perf_counter=perf, wall_now=wall, save_path=path,
                    save_lock=threading.Lock(),
                ))
            events = json_events(frames)
            self.assertEqual(events[-1],
                             {"event": "error", "message": "codec failed"})
            self.assertTrue(inner.closed)
            self.assertFalse(path.exists())
            self.assertNotIn("run_completed", [event["event"] for event in events])

    def test_infer_stream_construction_failure_emits_error_without_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            perf, wall = self.clocks()
            with contextlib.redirect_stderr(io.StringIO()):
                frames = list(stream_play.iter_stream_frames(
                    InferConstructionFailureTTS(), valid_request(),
                    synchronize=lambda: None, perf_counter=perf, wall_now=wall,
                    save_path=path, save_lock=threading.Lock(),
                ))
            events = json_events(frames)
            self.assertEqual(events[-1], {
                "event": "error", "message": "infer setup failed",
            })
            self.assertFalse(path.exists())
            self.assertNotIn("run_completed", [event["event"] for event in events])

    def test_synchronize_failure_emits_error_closes_iterator_and_skips_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            inner = CloseAwareIterator([np.array([0.25], dtype=np.float32)])
            perf, wall = self.clocks()
            with contextlib.redirect_stderr(io.StringIO()):
                frames = list(stream_play.iter_stream_frames(
                    CloseAwareTTS(inner), valid_request(),
                    synchronize=lambda: (_ for _ in ()).throw(
                        RuntimeError("sync failed")
                    ),
                    perf_counter=perf, wall_now=wall, save_path=path,
                    save_lock=threading.Lock(),
                ))
            events = json_events(frames)
            self.assertEqual(events[-1], {
                "event": "error", "message": "sync failed",
            })
            self.assertTrue(inner.closed)
            self.assertFalse(path.exists())
            self.assertNotIn("run_completed", [event["event"] for event in events])

    def test_wave_write_failure_emits_error_closes_iterator_and_skips_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            inner = CloseAwareIterator([np.array([0.25], dtype=np.float32)])
            perf, wall = self.clocks()
            with mock.patch.object(
                stream_play.sf, "write", side_effect=RuntimeError("write failed")
            ), contextlib.redirect_stderr(io.StringIO()):
                frames = list(stream_play.iter_stream_frames(
                    CloseAwareTTS(inner), valid_request(), synchronize=lambda: None,
                    perf_counter=perf, wall_now=wall, save_path=path,
                    save_lock=threading.Lock(),
                ))
            events = json_events(frames)
            self.assertEqual(events[-1], {
                "event": "error", "message": "write failed",
            })
            self.assertTrue(inner.closed)
            self.assertFalse(path.exists())
            self.assertNotIn("run_completed", [event["event"] for event in events])

    def test_closing_outer_iterator_closes_inner_iterator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            inner = CloseAwareIterator([np.array([0.25], dtype=np.float32)] * 3)
            perf, wall = self.clocks()
            outer = stream_play.iter_stream_frames(
                CloseAwareTTS(inner), valid_request(), synchronize=lambda: None,
                perf_counter=perf, wall_now=wall, save_path=path,
                save_lock=threading.Lock(),
            )
            next(outer)
            next(outer)
            outer.close()
            self.assertTrue(inner.closed)
            self.assertFalse(path.exists())

    def test_empty_audio_uses_null_ratios(self):
        perf = Sequence([0.0, 0.001, 0.002])
        wall = Sequence([
            datetime(2026, 9, 16, 14, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 1000, tzinfo=timezone.utc),
            datetime(2026, 9, 16, 14, 30, 0, 2000, tzinfo=timezone.utc),
        ])
        frames = list(stream_play.iter_stream_frames(
            FakeTTS(chunks=[]), valid_request(), synchronize=lambda: None,
            perf_counter=perf, wall_now=wall,
        ))
        completed = json_events(frames)[-1]
        self.assertIsNone(completed["first_chunk_ms"])
        self.assertIsNone(completed["rtf"])
        self.assertIsNone(completed["realtime_speed"])

    def test_successful_save_writes_clipped_48khz_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.wav"
            perf = Sequence([0.0, 0.001, 0.011, 0.012, 0.020])
            wall = Sequence([
                datetime(2026, 9, 16, 14, 30, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 1000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 11000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 12000, tzinfo=timezone.utc),
                datetime(2026, 9, 16, 14, 30, 0, 20000, tzinfo=timezone.utc),
            ])
            list(stream_play.iter_stream_frames(
                FakeTTS(chunks=[np.array([-2.0, 0.5, 2.0], dtype=np.float32)]),
                valid_request(), synchronize=lambda: None,
                perf_counter=perf, wall_now=wall, save_path=path,
                save_lock=threading.Lock(),
            ))
            audio, rate = sf.read(path, dtype="float32")
            self.assertEqual(rate, 48_000)
            np.testing.assert_allclose(audio, [-1.0, 0.5, 0.9999695], atol=1e-5)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(text="Mặc định", voice="Mai Anh")
        self.config = stream_play.build_config(
            FakeTTS(), args, hardware={"gpu": None}
        )

    def test_valid_request_is_normalized(self):
        value = valid_request()
        value["text"] = "  Xin chào  "
        self.assertEqual(
            stream_play.validate_stream_request(value, self.config)["text"],
            "Xin chào",
        )

    def test_invalid_values_raise_specific_value_errors(self):
        cases = [
            (None, "JSON object"),
            ({**valid_request(), "text": "   "}, "text"),
            ({**valid_request(), "text": "x" * 20001}, "20,000"),
            ({**valid_request(), "voice": "Unknown"}, "voice"),
            ({**valid_request(), "temperature": True}, "temperature"),
            ({**valid_request(), "temperature": float("inf")}, "temperature"),
            ({**valid_request(), "top_p": "0.95"}, "top_p"),
            ({**valid_request(), "top_k": 25.5}, "top_k"),
            ({**valid_request(), "top_k": 101}, "top_k"),
            ({**valid_request(), "top_p": 0.01}, "top_p"),
            ({**valid_request(), "repetition_penalty": 2.1}, "repetition_penalty"),
            ({**valid_request(), "max_chars": 127}, "max_chars"),
            ({**valid_request(), "max_new_frames": 0}, "max_new_frames"),
            ({**valid_request(), "max_new_frames": 1201}, "max_new_frames"),
            ({**valid_request(), "max_new_frames": 300.5}, "max_new_frames"),
            ({**valid_request(), "extra": 1}, "unknown field"),
            ({key: value for key, value in valid_request().items()
              if key != "top_k"}, "missing field"),
        ]
        for value, message in cases:
            with self.subTest(value=value, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    stream_play.validate_stream_request(value, self.config)


class TextChunkTest(unittest.TestCase):
    def test_text_chunk_records_match_vieneu_normalizer(self):
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

        text = "Câu thứ nhất đủ dài để kiểm tra. Câu thứ hai tiếp tục kiểm tra."
        expected_chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=128)
        records = stream_play.text_chunk_records(text, 128)
        self.assertEqual([r["text"] for r in records], expected_chunks)
        self.assertEqual([r["characters"] for r in records],
                         [len(chunk) for chunk in expected_chunks])
        self.assertEqual([r["index"] for r in records],
                         list(range(1, len(records) + 1)))

    def test_hardware_info_is_json_safe(self):
        json.dumps(stream_play.hardware_info(), allow_nan=False)


class HttpServerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ui_path = Path(self.tempdir.name) / "stream_ui.html"
        self.ui_path.write_text("<!doctype html><title>dashboard</title>", encoding="utf-8")
        self.args = SimpleNamespace(
            text="Văn bản mặc định", voice="Mai Anh", save=None,
            warmup=False, port=0, no_browser=True,
        )
        self.tts = FakeTTS()
        self.server = stream_play.create_server(
            self.tts, self.args, host="127.0.0.1", port=0,
            synchronize=lambda: None, hardware={"gpu": None},
            ui_path=self.ui_path,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = "127.0.0.1"
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tempdir.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        result = response.status, dict(response.getheaders()), data
        connection.close()
        return result

    def test_root_serves_external_ui_file(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(body, b"<!doctype html><title>dashboard</title>")

    def test_config_returns_json_schema(self):
        status, headers, body = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        config = json.loads(body)
        self.assertEqual(config["defaults"]["voice"], "Mai Anh")
        self.assertEqual(config["limits"]["top_k"]["max"], 100)

    def test_unknown_route_returns_404(self):
        status, _, _ = self.request("GET", "/missing")
        self.assertEqual(status, 404)

    def test_missing_ui_file_fails_during_server_creation(self):
        with self.assertRaisesRegex(FileNotFoundError, "missing.html"):
            stream_play.create_server(
                self.tts, self.args, port=0,
                synchronize=lambda: None, hardware={"gpu": None},
                ui_path=Path(self.tempdir.name) / "missing.html",
            )

    def test_server_rejects_non_localhost_binding(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            stream_play.create_server(
                self.tts, self.args, host="0.0.0.0", port=0,
                synchronize=lambda: None, hardware={"gpu": None},
                ui_path=self.ui_path,
            )

    def test_post_rejects_wrong_media_type(self):
        status, _, body = self.request(
            "POST", "/api/stream", b"{}", {"Content-Type": "text/plain"}
        )
        self.assertEqual(status, 415)
        self.assertIn("application/json", json.loads(body)["error"])

    def test_post_rejects_missing_and_oversized_lengths(self):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        connection.putrequest("POST", "/api/stream")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        connection.close()

        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        connection.putrequest("POST", "/api/stream")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "not-a-number")
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        connection.close()

        status, _, _ = self.request(
            "POST", "/api/stream", b"x",
            {"Content-Type": "application/json", "Content-Length": "65537"},
        )
        self.assertEqual(status, 400)

        status, _, body = self.request(
            "POST", "/api/stream", b"", {"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_post_rejects_invalid_content_length_syntax(self):
        body = json.dumps(valid_request()).encode()
        invalid_length = f"{len(body) // 10}_{len(body) % 10}"
        status, _, _ = self.request(
            "POST", "/api/stream", body,
            {"Content-Type": "application/json", "Content-Length": invalid_length},
        )
        self.assertEqual(status, 400)

    def test_post_rejects_incomplete_content_length_body(self):
        body = json.dumps(valid_request()).encode()
        request = (
            f"POST /api/stream HTTP/1.1\r\nHost: {self.host}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body) + 1}\r\nConnection: close\r\n\r\n"
        ).encode() + body
        with socket.create_connection((self.host, self.port), timeout=5) as connection:
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            response = connection.makefile("rb").read()
        self.assertTrue(response.startswith(b"HTTP/1.0 400"), response)

    def test_post_rejects_malformed_json_and_invalid_values(self):
        for body in (
            b"{",
            b"\xff",
            json.dumps({**valid_request(), "top_p": 2}).encode(),
        ):
            with self.subTest(body=body):
                status, headers, payload = self.request(
                    "POST", "/api/stream", body,
                    {"Content-Type": "application/json; charset=utf-8"},
                )
                self.assertEqual(status, 400)
                self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
                self.assertIn("error", json.loads(payload))

    def test_post_rejects_large_numeric_values_as_json_errors(self):
        body = json.dumps({**valid_request(), "top_k": 10 ** 400}).encode()
        status, headers, payload = self.request(
            "POST", "/api/stream", body,
            {"Content-Type": "application/json; charset=utf-8"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertIn("error", json.loads(payload))

    def test_post_rejects_deeply_nested_json_as_a_json_error(self):
        body = b"[" * 10_000 + b"0" + b"]" * 10_000
        status, headers, payload = self.request(
            "POST", "/api/stream", body,
            {"Content-Type": "application/json; charset=utf-8"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertIn("error", json.loads(payload))

    def test_header_disconnect_is_handled_without_handler_traceback(self):
        output = io.StringIO()
        errors = io.StringIO()
        body = json.dumps(valid_request()).encode()
        handler = self.server.RequestHandlerClass
        with mock.patch.object(handler, "end_headers", side_effect=BrokenPipeError), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
            connection.request(
                "POST", "/api/stream", body,
                {"Content-Type": "application/json; charset=utf-8"},
            )
            with self.assertRaises(http.client.RemoteDisconnected):
                connection.getresponse()
            connection.close()
        self.assertIn("Trình duyệt đã ngắt stream.", output.getvalue())
        self.assertNotIn("Exception occurred during processing", errors.getvalue())

    def test_valid_post_returns_ordered_framed_stream(self):
        body = json.dumps(valid_request(), ensure_ascii=False).encode()
        status, headers, payload = self.request(
            "POST", "/api/stream", body,
            {"Content-Type": "application/json; charset=utf-8"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/x-vieneu-stream")
        frames = decode_frames(payload)
        self.assertEqual([kind for kind, _ in frames], [1, 1, 2, 1, 2, 1])
        self.assertEqual(json.loads(frames[0][1])["event"], "run_started")
        self.assertEqual(json.loads(frames[-1][1])["event"], "run_completed")


class StartupTest(unittest.TestCase):
    def test_main_prints_the_actual_server_url_for_a_custom_port(self):
        args = SimpleNamespace(
            text="Văn bản mặc định", voice="Mai Anh", save=None,
            warmup=False, port=9137, no_browser=True,
        )
        server = mock.Mock(server_port=9137)
        server.serve_forever.side_effect = KeyboardInterrupt
        output = io.StringIO()
        with mock.patch.object(stream_play, "parse_args", return_value=args), \
                mock.patch.object(stream_play.torch.cuda, "is_available", return_value=True), \
                mock.patch.object(stream_play.torch.cuda, "get_device_name", return_value="Test GPU"), \
                mock.patch.object(stream_play, "Vieneu"), \
                mock.patch.object(stream_play, "create_server", return_value=server), \
                contextlib.redirect_stdout(output):
            stream_play.main()
        self.assertIn("Mở http://127.0.0.1:9137/ rồi cấu hình", output.getvalue())


if __name__ == "__main__":
    unittest.main()
