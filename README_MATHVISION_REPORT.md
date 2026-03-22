# Báo Cáo AI Nâng Cao - MathVision Thinking

README này tổng hợp cách chạy demo `mathvision_gradio_app.py`, thông tin model công khai trên Hugging Face, và link tài liệu báo cáo.

## 1) Thông tin chính

- Báo cáo PDF: [`docs/QwenReport.pdf`](docs/QwenReport.pdf)
- Hugging Face model: <https://huggingface.co/anhnq1130/mathvision-thinking>
- App demo: `mathvision_gradio_app.py`
- Script chạy nhanh: `run_mathvision_demo.sh`

## 2) Chức năng demo

Ứng dụng Gradio cho bài toán thị giác (MathVision):

- Nhập câu hỏi text
- Upload ảnh hoặc chụp webcam
- Hiển thị quá trình suy luận (thinking) và đáp án cuối
- Chuyển model ngay trên giao diện

## 3) Cài đặt môi trường

Khuyến nghị Python 3.10+ và có GPU CUDA để chạy nhanh.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gradio transformers peft pillow markdown accelerate
```

Nếu bạn đã có môi trường từ LlamaFactory thì có thể bỏ qua bước cài đặt thủ công.

## 4) Cách chạy app

### Cách 1: Chạy trực tiếp file Python

```bash
python mathvision_gradio_app.py --server-name 0.0.0.0 --server-port 7861
```

Mở trình duyệt tại:

- `http://127.0.0.1:7861` (chạy local)
- hoặc `http://<IP-máy-bạn>:7861` (truy cập trong LAN)

### Cách 2: Chạy bằng script có sẵn

```bash
chmod +x run_mathvision_demo.sh
./run_mathvision_demo.sh
```

Mặc định script dùng:

- `HOST=0.0.0.0`
- `PORT=7860`

Đặt biến môi trường nếu muốn đổi port/host:

```bash
HOST=0.0.0.0 PORT=7861 ./run_mathvision_demo.sh
```

Bật chia sẻ public của Gradio:

```bash
SHARE_FLAG=1 ./run_mathvision_demo.sh
```

## 5) Lưu ý về model Hugging Face của bạn

Model công khai của bạn:

- `anhnq1130/mathvision-thinking`

Trong file `mathvision_gradio_app.py`, phần `MODEL_SPECS` đang định nghĩa model thinking riêng cho demo. Nếu muốn app load trực tiếp model của bạn, cập nhật trường `adapter_name_or_path` hoặc `model_name_or_path` trong `MODEL_SPECS` thành:

```text
anhnq1130/mathvision-thinking
```

Sau đó chạy lại app.

## 6) Tài liệu báo cáo

Tài liệu chính của đề tài:

- [`docs/QwenReport.pdf`](docs/QwenReport.pdf)

## 7) Lỗi thường gặp

- Lỗi thiếu CUDA / VRAM: dùng model nhỏ hơn, tắt app khác dùng GPU, hoặc chạy CPU (sẽ chậm).
- Lỗi không load được adapter: kiểm tra tên repo Hugging Face, quyền public, và kết nối mạng.
- Port bị trùng: đổi `--server-port` (ví dụ `7862`).
