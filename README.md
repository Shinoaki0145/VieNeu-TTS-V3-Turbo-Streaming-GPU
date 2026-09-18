# Hướng dẫn chạy VieNeu Streaming Dashboard

Dashboard hỗ trợ phát giọng nói theo thời gian thực, theo dõi chi tiết hiệu
năng từng chunk, phát lại kết quả vừa tạo và lưu audio hoàn chỉnh thành file
WAV.

## 1. Yêu cầu

- Linux, WSL hoặc Windows.
- Python 3.12.
- PyTorch nhận được GPU NVIDIA và CUDA.
- Trình duyệt hỗ trợ Web Audio, ví dụ Chrome hoặc Edge.

### Tạo `.venv` trên Linux/WSL

```bash
git clone https://github.com/Shinoaki0145/VieNeu-TTS-V3-Turbo-Streaming-GPU.git
cd VieNeu-TTS-V3-Turbo-Streaming-GPU
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Nếu lệnh tạo môi trường báo thiếu module `venv`, cài bổ sung rồi chạy lại:

```bash
sudo apt update
sudo apt install python3.12-venv
```

### Tạo `.venv` trên Windows PowerShell

```powershell
git clone https://github.com/Shinoaki0145/VieNeu-TTS-V3-Turbo-Streaming-GPU.git
cd VieNeu-TTS-V3-Turbo-Streaming-GPU
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Cài thư viện cho GPU NVIDIA

Sau khi đã kích hoạt `.venv`, chọn cách cài phù hợp với hệ điều hành.

Trên Linux/WSL:

```bash
python -m pip install "vieneu[cuda]"
```

Trên Windows, cài PyTorch CUDA và các thư viện cần thiết:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "transformers==4.57.6"
python -m pip install vieneu
```

Kiểm tra Python hiện tại có nằm trong `.venv` hay không:

```bash
which python
python --version
```

Trên Linux/WSL, `which python` phải trả về đường dẫn kết thúc tương tự:

```text
.../VieNeu-TTS-V3-Turbo-Streaming-GPU/.venv/bin/python
```

Mỗi lần mở terminal Linux/WSL mới, kích hoạt lại môi trường bằng:

```bash
cd VieNeu-TTS-V3-Turbo-Streaming-GPU
source .venv/bin/activate
```

Các lệnh bên dưới được chạy sau khi `.venv` đã được tạo, cài thư viện và kích hoạt.

Kiểm tra GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không có')"
```

## 2. Chạy dashboard

Khuyến nghị warm-up GPU trước khi mở dashboard:

```bash
python stream_play.py --warmup
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
4. Nhấn **Phát** để bắt đầu stream.
5. Sau khi toàn bộ stream hoàn tất, hai nút sau sẽ được mở khóa:
   - **Phát lại**: nghe lại audio hoàn chỉnh mà không chạy model thêm lần nữa.
   - **Lưu**: ghi audio thành file WAV trong thư mục `output/` trên máy đang chạy server.
6. Nhấn **Dừng** để huỷ request, dừng audio và xoá audio hoàn chỉnh đang giữ tạm.
7. Nhấn **Đặt lại** để trả các trường về cấu hình ban đầu và xoá audio tạm.

Trong lúc stream, audio vẫn được phát theo từng chunk như bình thường. Nút
**Phát lại** và **Lưu** chỉ dùng được sau khi request đã hoàn tất thành công.
Nếu request đang chạy, bị dừng hoặc gặp lỗi, hai nút này sẽ bị khóa.

File chỉ được ghi xuống ổ đĩa sau khi nhấn **Lưu**. Đây là thao tác lưu trên
máy chạy `stream_play.py`, không phải tải file bằng trình quản lý tải xuống
của trình duyệt. Nếu thư mục `output/` chưa tồn tại, chương trình sẽ tự tạo.
Tên file gồm 14 ký tự chữ và số ngẫu nhiên, sau đó là đuôi `.wav`, ví dụ
`Ab3xY7kL9mN2qR.wav`. Sau khi lưu thành công, giao diện sẽ hiển thị đường dẫn
của file.

Dashboard chỉ giữ audio hoàn chỉnh gần nhất. Khi bắt đầu stream mới, nhấn
**Dừng**, nhấn **Đặt lại** hoặc inference gặp lỗi, audio tạm trước đó sẽ không
còn khả dụng để phát lại hoặc lưu. Sau khi lưu thành công, nút **Lưu** được
khóa để tránh tạo nhiều bản trùng.

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
- Nhấn trực tiếp **Phát** để trình duyệt cho phép khởi tạo Web Audio.
- Kiểm tra tab không bị mute và Windows đang chọn đúng thiết bị phát.
- Không mở trực tiếp file `stream_ui.html`; phải truy cập qua địa chỉ server `http://127.0.0.1:<port>/`.

### Nút Phát lại hoặc Lưu đang bị khóa

- Chờ stream hiện tại chạy xong hoàn toàn. Hai nút chỉ được bật sau khi nhận được kết quả hoàn chỉnh.
- Không nhấn **Dừng** hoặc **Đặt lại**, vì các thao tác này sẽ xoá audio đang giữ tạm.
- Nếu inference báo lỗi, hãy chạy lại bằng nút **Phát**.

### Không tìm thấy file WAV đã lưu

Sau khi nhấn **Lưu**, kiểm tra thư mục `output/` trong thư mục repo. Giao diện
cũng hiển thị đường dẫn file ngay sau khi lưu thành công. Nếu server chạy trên
một máy khác, file nằm trên máy chạy server chứ không nằm trên máy đang mở
trình duyệt.

### Model tải lần đầu lâu

Đây là bình thường nếu model hoặc cache chưa sẵn có. Dùng `--warmup` để lần phát đầu tiên ổn định hơn.
