# DoCaSeg (Dog & Cat Segmentation)

DoCaSeg là mô hình phân đoạn ảnh ngữ nghĩa (semantic segmentation) dựa trên kiến trúc **UNet** áp dụng trên tập dữ liệu **Oxford-IIIT Pet Dataset** để phân đoạn chó mèo thành 3 vùng:
- **Foreground (Lớp 0)**: Cơ thể thú cưng
- **Background (Lớp 1)**: Nền xung quanh
- **Boundary (Lớp 2)**: Đường viền/contour

---

## 📁 Cấu trúc dự án

```text
DoCaSeg/
├── configs/
│   └── config.yaml          # Cấu hình siêu tham số mô hình & huấn luyện
├── data/                    # Thư mục lưu dữ liệu Oxford-IIIT Pet (tự động tải)
├── checkpoints/             # Lưu trữ trọng số mô hình tốt nhất (best_model.pth)
├── outputs/                 # Thư mục lưu kết quả trực quan hóa dự đoán
│   └── predictions/
├── src/
│   ├── data/                # Dataset & DataLoader loader cho Oxford-IIIT Pet
│   │   └── dataset.py
│   ├── models/              # Kiến trúc UNet và các block
│   │   ├── blocks.py        # ConvBlock, Encoder, Decoder
│   │   └── model.py         # Lớp mạng UNet
│   ├── training/            # Vòng lặp đánh giá mô hình
│   │   └── evaluator.py
│   ├── evaluation/          # Các độ đo phân đoạn (Pixel Acc, mIoU, Dice)
│   │   └── metrics.py
│   └── utils/               # Tiện ích bổ trợ (seed, visualize)
│       ├── seed.py
│       └── visualization.py
├── tests/
│   └── test_model.py        # Unit test kiến trúc mô hình và độ đo
├── train.py                 # Script huấn luyện mô hình
├── evaluate.py              # Script đánh giá checkpoint trên tập test
├── predict.py               # Script phân đoạn ảnh mới
├── requirements.txt         # Danh sách thư viện cần thiết
└── README.md
```

---

## ⚙️ Cài đặt môi trường

Kích hoạt môi trường conda đã có sẵn:
```bash
conda activate basic_ml_cpu
```

Hoặc cài đặt các thư viện cần thiết từ `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)
Chạy script huấn luyện với các tham số mặc định (20 epochs, batch size 32):
```bash
python train.py
```
Tùy chỉnh tham số khi chạy:
```bash
python train.py --epochs 25 --batch-size 32 --lr 0.001 --data-root data/pets_data
```
Trọng số tốt nhất dựa trên validation loss sẽ được tự động lưu tại `checkpoints/best_model.pth`.

### 2. Đánh giá mô hình (Evaluation)
Đánh giá trọng số đã lưu trên tập test:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```
Kết quả hiển thị bao gồm:
- **Loss** (Cross-Entropy Loss)
- **Pixel Accuracy**
- **Mean IoU (mIoU)**
- **Dice Score**

### 3. Dự đoán trên ảnh bất kỳ (Inference)
Chạy phân đoạn trên một file ảnh và lưu kết quả:
```bash
python predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth --output outputs/predictions/result.png
```

### 4. Chạy Unit Tests
Kiểm tra tính đúng đắn của các module mạng:
```bash
pytest -v tests/test_model.py
```
*(Nếu môi trường có biến ROS PYTHONPATH, chạy: `PYTHONPATH="" pytest -v tests/test_model.py`)*

