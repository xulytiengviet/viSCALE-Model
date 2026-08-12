# 🦟 viSCALE-Model — Phát hiện và định vị muỗi bằng học sâu

[![Python](https://img.shields.io/badge/Python-3.9%20khuyến%20nghị-blue?logo=python)](https://www.python.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-Faster%20R--CNN-orange)](https://github.com/facebookresearch/detectron2)
[![Ngôn ngữ](https://img.shields.io/badge/Tài%20liệu-Tiếng%20Việt-red)](README.en.md)

> **viSCALE-Model** là bản Việt hóa chuyên nghiệp của kho mã mô hình SCALE, phục vụ nghiên cứu, học tập và tái lập quy trình phát hiện muỗi bằng thị giác máy tính. Bản Việt hóa giữ nguyên thuật toán, cấu trúc dữ liệu, tên API, checkpoint và luồng huấn luyện; chỉ chuẩn hóa tài liệu, chú thích và thông báo dành cho người dùng.

📘 **English:** [README.en.md](README.en.md)

---

## 1. Giới thiệu

Kho mã này thuộc dự án luận văn **“SCALE: Mosquito Detection and Fumigation System Utilizing Convolutional Neural Networks”** — hệ thống phát hiện muỗi và hỗ trợ phun khử bằng mạng nơ-ron tích chập.

Mục tiêu của dự án là xây dựng mô hình học sâu có khả năng:

- phát hiện muỗi trong ảnh;
- xác định vị trí của từng cá thể bằng hộp giới hạn (*bounding box*);
- hỗ trợ nhận diện khu vực có mật độ muỗi cao;
- cung cấp cơ sở dữ liệu và đầu ra cho các bước phân tích, giám sát hoặc hỗ trợ phun khử;
- xây dựng quy trình đầy đủ từ thu thập dữ liệu, gán nhãn, huấn luyện đến đánh giá mô hình.

🌐 Website của dự án gốc: [scale-anti-mosquito.site](https://www.scale-anti-mosquito.site)

### Các kho mã liên quan của dự án gốc

- [SCALE Backend](https://github.com/PeteCastle/SCALE-Backend)
- [SCALE Frontend](https://github.com/PeteCastle/SCALE-Frontend)
- [Mã Arduino của nguyên mẫu SCALE](https://github.com/PeteCastle/SCALE-Arduino)

---

## 2. Công nghệ chính

| Thành phần | Vai trò |
|---|---|
| **Python 3.9** | Môi trường chạy được khuyến nghị |
| **PyTorch** | Nền tảng học sâu |
| **Detectron2** | Huấn luyện và đánh giá mô hình phát hiện đối tượng |
| **Faster R-CNN** | Kiến trúc phát hiện đối tượng chính |
| **ResNet-50** | Backbone trích xuất đặc trưng |
| **FPN** | Trích xuất đặc trưng đa tỉ lệ |
| **CVAT** | Gán nhãn dữ liệu ảnh |
| **COCO JSON** | Định dạng chú thích dùng cho huấn luyện/đánh giá |
| **Jupyter Notebook** | Thử nghiệm, huấn luyện và phân tích |
| **SORT** | Theo dõi đối tượng theo thời gian trong một số luồng xử lý |

---

## 3. Cài đặt

### 3.1. Tạo môi trường Python

Khuyến nghị sử dụng **Python 3.9** và một môi trường ảo riêng.

Ví dụ với `venv`:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```bash
.venv\Scripts\activate
```

Kích hoạt trên Linux/macOS:

```bash
source .venv/bin/activate
```

### 3.2. Cài phụ thuộc

```bash
pip install -r requirements.txt
```

### 3.3. Cài PyTorch tương thích CUDA 11.3

Lệnh dưới đây được giữ theo cấu hình của dự án gốc:

```bash
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

> Cấu hình trên yêu cầu thiết bị hỗ trợ CUDA. Nếu sử dụng CPU hoặc phiên bản CUDA khác, cần chọn bộ PyTorch tương thích với môi trường thực tế.

### 3.4. Huấn luyện

Mở và chạy notebook:

```text
src/train.ipynb
```

Kết quả của mô hình được lưu trong thư mục:

```text
output/
```

---

## 4. Tổng quan kiến trúc

![Kiến trúc phần mềm tập trung vào học máy](assets/scale_software_arch_ml_focused.png)

Kiến trúc học máy được chia thành hai giai đoạn chính: **huấn luyện** và **kiểm thử**.

### Giai đoạn huấn luyện

Hệ thống tiếp nhận ảnh huấn luyện cùng tệp chú thích được tạo bằng CVAT và xuất theo định dạng COCO. Dữ liệu đi qua các bước xử lý ảnh gồm:

1. thay đổi kích thước (*resizing*);
2. chuẩn hóa (*normalization*);
3. tăng cường dữ liệu (*data augmentation*);
4. sinh vùng đề xuất (*region proposal*);
5. trích xuất đặc trưng;
6. phát hiện đối tượng bằng Faster R-CNN trên Detectron2.

Mô hình sử dụng **ResNet-50** làm backbone, kết hợp **Feature Pyramid Network (FPN)**, **Region Proposal Network (RPN)**, ROI pooling và các lớp kết nối đầy đủ.

### Giai đoạn kiểm thử

Ảnh kiểm thử được tiền xử lý tương tự dữ liệu huấn luyện. Mô hình thực hiện:

- ROI pooling;
- phân loại đối tượng;
- hồi quy hộp giới hạn (*bounding-box regression*);
- loại bỏ dự đoán trùng lặp bằng Non-Maximum Suppression (NMS).

Đầu ra gồm:

- hộp giới hạn;
- nhãn lớp;
- điểm tin cậy.

Các dự đoán được so sánh với dữ liệu chuẩn (*ground truth*) để tính các chỉ số đánh giá hiệu năng.

---

## 5. Dữ liệu và gán nhãn

### 5.1. Thu thập dữ liệu

Dự án gốc sử dụng:

- **20 ảnh thô** chứa nhiều muỗi, thu thập từ các nguồn công khai qua Google Images;
- **1.044 ảnh cá thể** và các tập dữ liệu của ba loài muỗi:
  - *Aedes aegypti L.*;
  - *Aedes albopictus L.*;
  - *Culex quinquefasciatus*;
- nguồn dữ liệu được dự án gốc dẫn theo Ong & Ahmad (2022) và Pise, Patil, Laad & Pise (2022).

### 5.2. Tăng cường dữ liệu

Hai mươi ảnh chứa nhiều muỗi được tăng cường bằng nhiều phép biến đổi, bao gồm:

- thay đổi kích thước;
- lật ngang/lật dọc;
- cắt ngẫu nhiên;
- Gaussian blur;
- thay đổi màu;
- chuẩn hóa độ tương phản;
- thêm nhiễu Gaussian.

Từ 20 ảnh ban đầu, dự án tạo ra **300 ảnh tăng cường**, sau đó kết hợp với 1.044 ảnh cá thể từ các tập dữ liệu hiện có.

### 5.3. Gán nhãn dữ liệu

![Quy trình gán nhãn bằng CVAT.AI](assets/data_annotation_process.png)

Dự án sử dụng **CVAT — Computer Vision Annotation Tool** để gán nhãn ảnh. CVAT hỗ trợ nhiều dạng chú thích như:

- bounding box;
- polygon;
- keypoint.

Việc gán nhãn được thực hiện thủ công, chia đều dữ liệu cho các thành viên nhằm duy trì tính nhất quán và phân bổ khối lượng công việc.

### 5.4. Chia tập huấn luyện/kiểm thử

Dữ liệu được chia theo tỉ lệ **70% huấn luyện / 30% kiểm thử**.

| Nhóm dữ liệu | Huấn luyện | Kiểm thử |
|---|---:|---:|
| Ảnh chứa nhiều muỗi | 210 | 90 |
| Ảnh cá thể | 744 | 300 |

---

## 6. Giai đoạn huấn luyện

Dự án sử dụng **Detectron2**, nền tảng phát hiện đối tượng mã nguồn mở do Facebook AI Research phát triển. Detectron2 hỗ trợ nhiều tác vụ và kiến trúc như:

- Faster R-CNN;
- Mask R-CNN;
- Cascade R-CNN;
- bounding box xoay;
- phân đoạn thể hiện (*instance segmentation*);
- phân đoạn ngữ nghĩa;
- phát hiện keypoint người.

![Detectron2 sử dụng họ kiến trúc Region-based CNN](assets/detectron2.png)

---

## 7. Backbone Network

Vai trò chính của backbone là nhận ảnh đầu vào và tạo ra các **feature map**.

Ảnh đầu vào được biểu diễn dưới dạng tensor có cấu trúc:

```text
(Batch size, 3 kênh BGR, Chiều cao ảnh, Chiều rộng ảnh)
```

Đầu ra của backbone là một tập tensor đặc trưng ở nhiều tỉ lệ. Thông thường số kênh đặc trưng mặc định là **256**, với stride:

| Mức | Stride |
|---|---:|
| P2 | 4 |
| P3 | 8 |
| P4 | 16 |
| P5 | 32 |
| P6 | 64 |

Các mức P2–P6 biểu diễn thông tin ở nhiều độ phân giải và *receptive field* khác nhau. Đây là yếu tố quan trọng để phát hiện các đối tượng có kích thước khác nhau.

---

## 8. Feature Pyramid Network — FPN

### 8.1. ResNet

ResNet gồm các khối *stem* và các *stage*. Khối stem giảm kích thước không gian của ảnh và tạo feature map ban đầu. Trong ResNet-50, kích thước đầu ra sau stem nhỏ hơn đáng kể so với ảnh đầu vào.

### 8.2. Bottleneck Block

Bottleneck block gồm ba lớp tích chập với các kích thước kernel khác nhau. Thiết kế này giúp giảm chi phí tính toán bằng cách giảm số kênh ở lớp tích chập trung gian.

### 8.3. Shortcut Connection

ResNet sử dụng kết nối tắt (*shortcut/residual connection*) để cộng đặc trưng đầu vào với đầu ra. Một số khối dùng convolution với `stride=2` để giảm kích thước feature map và đồng bộ số kênh.

### 8.4. FPN

FPN kết hợp:

- ResNet;
- lateral convolution;
- output convolution;
- up-sampling;
- lớp max-pooling cuối.

Các lateral convolution lấy đặc trưng từ nhiều tầng ResNet và chuyển chúng thành feature map 256 kênh.

### 8.5. Luồng truyền xuôi trong FPN

Từ đầu ra `res5`, FPN tạo P5 ở tỉ lệ **1/32**. P5 được upsample và kết hợp với `res4` để tạo P4 ở tỉ lệ **1/16**. Quy trình tiếp tục để tạo:

- P2: 1/4;
- P3: 1/8;
- P4: 1/16;
- P5: 1/32.

### 8.6. LastLevelMaxPool

P6 được tạo bằng cách giảm mẫu P5 từ tỉ lệ 1/32 xuống **1/64** bằng max pooling.

---

## 9. Ground Truth

Trong huấn luyện mô hình phát hiện đối tượng, *ground truth* chứa thông tin về vị trí và lớp của đối tượng. Với Faster R-CNN/FPN, dữ liệu này được sử dụng trong RPN và Box Head.

![Ví dụ bounding box và nhãn danh mục trong JSON](assets/ground_truth_bounding_box.png)

Một chú thích phát hiện đối tượng thường gồm hai thành phần chính:

1. **Nhãn hộp (box label):** mô tả vị trí/kích thước đối tượng, thường có dạng `[x, y, width, height]` hoặc tọa độ góc hộp.
2. **Nhãn danh mục (category label):** mã lớp của đối tượng. Trong ví dụ của dự án gốc, category ID `1` đại diện cho *Aedes aegypti*.

### Quy trình nạp ground truth

1. Đọc tập dữ liệu từ tệp JSON và tạo danh sách bản ghi ảnh/chú thích.
2. Dataset mapper bổ sung dữ liệu ảnh và chú thích `Instances` vào cấu trúc dữ liệu.
3. Ảnh được đọc, biến đổi và chuyển thành tensor.
4. Chú thích được chuyển thành `Instances`, gồm tọa độ bounding box và category ID ở dạng tensor.

---

## 10. Region Proposal Network — RPN

RPN kết nối feature map với vị trí và kích thước của các hộp ground truth. Nó gồm một **RPN Head** và các thành phần sinh vùng ứng viên.

Luồng xử lý chính:

1. Sinh anchor và liên kết objectness map/anchor-delta với anchor box.
2. Tính ma trận **Intersection over Union (IoU)** giữa ground-truth box và anchor.
3. Gán anchor thành foreground, background hoặc ignored theo ngưỡng IoU.
4. Tính hai nhóm loss chính:
   - localization loss;
   - objectness loss.
5. Áp dụng anchor delta dự đoán lên anchor.
6. Xếp hạng proposal theo objectness score.
7. Chọn các proposal có điểm cao và áp dụng **Non-Maximum Suppression**.

Theo mô tả của dự án gốc, hệ thống giữ khoảng **1.000 region proposal** cho các bước tiếp theo.

---

## 11. ROI / Box Head

ROI Head là giai đoạn cuối của pipeline phát hiện đối tượng. Đầu vào gồm:

- feature map từ FPN;
- proposal box từ RPN;
- ground-truth box khi huấn luyện.

Quy trình:

1. Nhận feature map ở nhiều mức FPN.
2. Sử dụng `proposal_boxes` từ RPN để xác định vùng quan tâm.
3. Thực hiện ROI pooling trên các vùng tương ứng.
4. Chuyển đặc trưng đã crop qua Box Head.
5. Các lớp cuối tạo:
   - `cls_score` cho phân loại;
   - `bbox_pred` cho hồi quy bounding box.

Đầu ra là tensor chứa điểm lớp và dự đoán bounding box cho từng ROI.

![Ví dụ tensor đầu ra của các đối tượng muỗi được phát hiện](assets/detected_mosquitos.png)

---

## 12. Kiểm thử và đánh giá mô hình

Detectron2 cung cấp công cụ đánh giá tích hợp. Hai nhóm chỉ số chính là **Average Precision (AP)** và **Average Recall (AR)**.

### 12.1. Average Precision — AP

Thường bao gồm:

- AP trung bình trên các ngưỡng IoU từ 0.50 đến 0.95;
- AP tại IoU = 0.50;
- AP tại IoU = 0.75;
- AP cho đối tượng nhỏ;
- AP cho đối tượng trung bình;
- AP cho đối tượng lớn.

### 12.2. Average Recall — AR

Thường bao gồm:

- AR trên dải IoU 0.50–0.95 với tối đa 1 detection/ảnh;
- AR với tối đa 10 detection/ảnh;
- AR với tối đa 100 detection/ảnh;
- AR cho đối tượng nhỏ;
- AR cho đối tượng trung bình;
- AR cho đối tượng lớn.

Ví dụ chỉ số COCO:

```text
AR @[IoU=0.50:0.95 | area=small | maxDets=100]
```

Dự án gốc thử nghiệm nhiều cấu hình Faster R-CNN:

- R50-FPN (1x learning rate);
- R50-FPN (3x learning rate);
- R101-FPN;
- X101-FPN.

Mô hình tốt nhất được lựa chọn chủ yếu dựa trên:

1. Average Precision cho đối tượng lớn;
2. Average Precision trung bình trên dải IoU 0.50–0.95.

---

## 13. Cấu trúc kho mã

```text
viSCALE-Model/
├── assets/                  # Hình minh họa kiến trúc và tài liệu
├── datasets/                # Dữ liệu/tệp hỗ trợ dữ liệu
├── src/
│   ├── core/
│   │   └── sort.py          # Bộ theo dõi SORT
│   ├── utils/
│   │   ├── augment_data.ipynb
│   │   ├── convert.ipynb
│   │   ├── group_data.ipynb
│   │   ├── read_training_log.ipynb
│   │   └── voc2coco.py      # Chuyển Pascal VOC XML → COCO JSON
│   ├── frcnn_test_vgg.ipynb
│   ├── labels.txt
│   └── train.ipynb          # Notebook huấn luyện chính
├── requirements.txt
├── README.en.md             # Tài liệu tiếng Anh được bảo tồn
└── readme.md                # Tài liệu tiếng Việt
```

---

## 14. Nguyên tắc Việt hóa

Bản Việt hóa tuân theo các nguyên tắc sau:

- **Không đổi thuật toán** và không chỉnh tham số mô hình chỉ vì mục đích dịch thuật.
- **Không dịch tên API, class, function, key JSON, tensor name hoặc checkpoint** nếu việc dịch có thể làm hỏng khả năng chạy lại.
- Giữ các thuật ngữ quốc tế quan trọng như `bounding box`, `IoU`, `ROI`, `RPN`, `FPN`, `NMS`, `ground truth` kèm giải thích tiếng Việt.
- Chỉ dịch chuỗi giao diện, thông báo dòng lệnh, docstring và tài liệu dành cho con người.
- Giữ nguyên thông tin bản quyền/giấy phép của mã nguồn bên thứ ba.

---

## 15. Lưu ý về giấy phép và nguồn gốc

Kho mã hiện không có tệp `LICENSE` ở thư mục gốc. Vì vậy không nên suy diễn một giấy phép chung cho toàn bộ dự án nếu chưa có xác nhận từ tác giả gốc.

Riêng `src/core/sort.py` chứa thông báo bản quyền và giấy phép GNU GPL của tác giả SORT; phần thông báo pháp lý này được **giữ nguyên nguyên văn** trong bản Việt hóa.

Bản Việt hóa không thay đổi quyền tác giả của dự án gốc, dữ liệu gốc hoặc các thành phần mã nguồn bên thứ ba.

---

## 16. Gợi ý sử dụng trong nghiên cứu và giáo dục

Repo phù hợp để nghiên cứu/thực hành các nội dung:

- phát hiện đối tượng bằng Faster R-CNN;
- học chuyển giao với backbone ResNet;
- Feature Pyramid Network;
- gán nhãn dữ liệu bằng CVAT;
- chuyển đổi Pascal VOC sang COCO;
- đánh giá mô hình theo AP/AR và IoU;
- xây dựng hệ thống thị giác máy tính phục vụ giám sát côn trùng truyền bệnh.

---

## 17. Ghi nhận dự án gốc

Mọi nội dung thuật toán, kiến trúc, dữ liệu mô tả và hình minh họa cốt lõi thuộc dự án SCALE gốc và các nguồn mà dự án đã dẫn.

Bản `viSCALE-Model` tập trung vào **Việt hóa tài liệu và trải nghiệm sử dụng**, giúp người học và nhà phát triển Việt Nam tiếp cận quy trình dễ dàng hơn mà không làm thay đổi bản chất kỹ thuật của mô hình.

![Footer dự án SCALE](assets/petecastle_footer.png)
