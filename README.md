# Hướng dẫn chạy VieNeu Streaming Dashboard

## 1. Yêu cầu

- Chạy trong thư mục `my_test` trên Linux, WSL hoặc Windows.
- Môi trường Python đã tồn tại tại `my_test/.venv` và được cài đầy đủ thư viện.
- PyTorch nhận được GPU NVIDIA và CUDA.
- Trình duyệt hỗ trợ Web Audio, ví dụ Chrome hoặc Edge.

Kích hoạt môi trường trên Linux/WSL:

```bash
cd /duong/dan/toi/VieNeu-TTS/my_test
source .venv/bin/activate
```

Kích hoạt môi trường trên Windows PowerShell:

```powershell
cd C:\duong\dan\toi\VieNeu-TTS\my_test
.venv\Scripts\Activate.ps1
```

Các lệnh bên dưới được chạy sau khi `.venv` đã được kích hoạt.

Kiểm tra GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không có')"
```

## 2. Chạy dashboard

Khuyến nghị chạy warm-up GPU và lưu kết quả cuối ra WAV:

```bash
python stream_play.py --warmup --save output_stream.wav
```

Sau khi model tải xong, chương trình sẽ mở địa chỉ:

```text
http://127.0.0.1:8001/
```

Nếu trình duyệt không tự mở, hãy sao chép địa chỉ trên và mở bằng Chrome hoặc Edge trên Windows.

## 3. Sử dụng giao diện

Trên dashboard:

1. Nhập nội dung tiếng Việt cần đọc.
2. Chọn giọng đọc.
3. Điều chỉnh các tham số nếu cần:
   - `temperature`: độ ngẫu nhiên của giọng đọc.
   - `top_k`: số lựa chọn token ưu tiên.
   - `top_p`: ngưỡng xác suất tích luỹ.
   - `repetition_penalty`: hạn chế lặp âm hoặc lặp nội dung.
   - `max_chars`: số ký tự tối đa của mỗi text chunk.
   - `max_new_frames`: số frame audio tối đa được sinh cho mỗi chunk (`1`–`1200`, mặc định `300`).
4. Nhấn **Phát stream**.
5. Nhấn **Dừng** để huỷ request và dừng audio đang phát.
6. Nhấn **Khôi phục mặc định** để trả các trường về cấu hình ban đầu.

Trong lúc chạy, giao diện hiển thị:

- Cấu hình CPU, RAM, GPU, VRAM, CUDA, Torch, Python và VieNeu.
- Chunk đầu tiên / TTFA.
- Audio duration, generation time, RTF và realtime speed.
- Danh sách text chunk cùng số ký tự.
- Danh sách audio chunk cùng samples, thời lượng và các mốc thời gian.

## 4. Các tuỳ chọn dòng lệnh

Xem đầy đủ trợ giúp:

```bash
python stream_play.py --help
```

Ví dụ thay port:

```bash
python stream_play.py --warmup --port 9000
```

Sau đó mở:

```text
http://127.0.0.1:9000/
```

Chạy nhưng không tự mở trình duyệt:

```bash
python stream_play.py --warmup --no-browser
```

Đặt text và giọng mặc định khi khởi động:

```bash
python stream_play.py \
  --text "Xin chào, đây là nội dung kiểm tra." \
  --voice "Mai Anh"
```

Trên Windows PowerShell, có thể viết cùng lệnh trên một dòng:

```powershell
python stream_play.py --text "Xin chào, đây là nội dung kiểm tra." --voice "Mai Anh"
```

Lưu audio sau mỗi request thành công:

```bash
python stream_play.py --save output_stream.wav
```

File WAV chỉ được ghi sau khi request hoàn tất. Nếu nhấn **Dừng** hoặc inference gặp lỗi, chương trình không ghi file audio dở.

## 5. Dừng server

Quay lại terminal đang chạy chương trình và nhấn:

```text
Ctrl+C
```

## 6. Chạy kiểm thử

```bash
python -m unittest -v test_stream_play.py
python -m py_compile stream_play.py test_stream_play.py
```

## 7. Xử lý lỗi thường gặp

### CUDA không khả dụng

Nếu xuất hiện thông báo `CUDA không khả dụng`, kiểm tra lại NVIDIA driver trên Windows, CUDA trong WSL và môi trường Python đang sử dụng.

### Port đang được sử dụng

Chọn port khác:

```bash
python stream_play.py --port 9000
```

### Có telemetry nhưng không nghe được audio

- Đảm bảo dashboard được mở bằng trình duyệt Windows, không phải trình duyệt Linux trong WSL.
- Nhấn trực tiếp **Phát stream** để trình duyệt cho phép khởi tạo Web Audio.
- Kiểm tra tab không bị mute và Windows đang chọn đúng thiết bị phát.
- Không mở trực tiếp file `stream_ui.html`; phải truy cập qua địa chỉ server `http://127.0.0.1:<port>/`.

### Model tải lần đầu lâu

Đây là bình thường nếu model hoặc cache chưa sẵn có. Dùng `--warmup` để lần phát đầu tiên ổn định hơn.
