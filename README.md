# Traffic Sign Recognition

Dự án xây dựng hệ thống nhận diện biển báo giao thông bằng Computer Vision và Deep Learning.

Mục tiêu của project là xây dựng pipeline hoàn chỉnh: dataset → preprocessing → training → evaluation → demo.

Nguyên tắc làm project:
- Ưu tiên dùng thư viện có sẵn.
- Không tự code lại thuật toán nếu thư viện đã hỗ trợ.
- Dùng TensorFlow/Keras để xây dựng và huấn luyện mô hình.
- Dùng scikit-learn để đánh giá mô hình.
- Dùng OpenCV/Pillow để xử lý ảnh.
- Dùng Streamlit hoặc Gradio để làm demo.
- Kết quả cần có bảng, biểu đồ, report để dễ đưa vào paper/báo cáo.

---

## 1. Python Version

Khuyến nghị dùng Python 3.10 hoặc Python 3.11.

Môi trường hiện tại của project đang dùng Python 3.11.0.

Lưu ý: Không nên dùng Python 3.13 cho project này vì TensorFlow có thể lỗi tương thích.

---

## 2. Cấu trúc project

traffic-sign-recognition/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── sample_images/
│       └── README.md
│
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_baseline.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_evaluation_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train_cnn.py
│   ├── train_transfer.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app/
│   ├── streamlit_app.py
│   └── gradio_app.py
│
├── models/
│   ├── README.md
│   ├── cnn_baseline.keras
│   └── transfer_model.keras
│
├── results/
│   ├── figures/
│   │   ├── cnn_accuracy_loss.png
│   │   ├── transfer_accuracy_loss.png
│   │   └── confusion_matrix.png
│   │
│   ├── reports/
│   │   ├── cnn_classification_report.txt
│   │   ├── transfer_classification_report.txt
│   │   └── comparison_table.csv
│   │
│   └── predictions/
│       └── sample_predictions.csv
│
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── check_env.py

Ghi chú: Các file model, hình ảnh kết quả, classification report có thể chưa có ở Week 0. Chúng sẽ được sinh ra sau khi train/evaluate model.

---

## 3. Ý nghĩa các thư mục chính

data/
- Chứa dataset.
- raw/ chứa dữ liệu gốc tải về.
- processed/ chứa dữ liệu đã chia train/val/test.
- sample_images/ chứa ảnh test ngoài dataset để demo.

notebooks/
- Chứa notebook thí nghiệm.
- Dùng để khám phá dữ liệu, train thử model và đánh giá kết quả.

src/
- Chứa source code chính của project.
- Các file train/evaluate/predict có thể phát triển dần từ Week 1 trở đi.

app/
- Chứa code demo.
- streamlit_app.py dùng để chạy demo bằng Streamlit.
- gradio_app.py dùng để chạy demo bằng Gradio.

models/
- Chứa model đã train.
- Ví dụ: cnn_baseline.keras, transfer_model.keras.

results/
- Chứa kết quả huấn luyện và đánh giá.
- figures/ chứa biểu đồ.
- reports/ chứa classification report và bảng so sánh.
- predictions/ chứa kết quả dự đoán ảnh mẫu.

---

## 4. Các thư viện sử dụng

Các thư viện chính:
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- opencv-python
- pillow
- streamlit
- gradio
- jupyter
- notebook
- ipykernel

Ý nghĩa:
- numpy: xử lý dữ liệu dạng mảng số.
- pandas: đọc và xử lý file CSV/label.
- matplotlib: vẽ biểu đồ accuracy/loss.
- scikit-learn: đánh giá model bằng metrics.
- tensorflow: xây dựng và train model deep learning.
- opencv-python: đọc, resize và xử lý ảnh.
- pillow: mở ảnh, đặc biệt khi upload ảnh trong demo.
- streamlit: làm demo web app.
- gradio: làm demo AI nhanh.
- jupyter/notebook/ipykernel: chạy notebook thí nghiệm.

Lưu ý:
- Không cần cài riêng sklearn.metrics vì nó nằm trong scikit-learn.
- Không cần cài riêng ImageDataGenerator vì nó nằm trong TensorFlow/Keras.
- Không cần cài riêng image_dataset_from_directory vì nó nằm trong TensorFlow/Keras.
- Không cần cài riêng keras nếu dùng tensorflow.keras.

---

## 5. Setup môi trường

Mở terminal tại thư mục project.

Ví dụ project nằm ở D:\School\traffic-sign-recognition thì chạy:

cd /d D:\School\traffic-sign-recognition

Tạo virtual environment:

python -m venv .venv

Kích hoạt virtual environment trên Windows cmd:

.venv\Scripts\activate

Nếu kích hoạt thành công, terminal sẽ hiện dạng:

(.venv) D:\School\traffic-sign-recognition>

Kiểm tra Python đang dùng đúng venv chưa:

where python

Dòng đầu tiên nên là:

D:\School\traffic-sign-recognition\.venv\Scripts\python.exe

Nếu dòng đầu tiên là Python trong .venv thì đúng.

---

## 6. Cài thư viện

Sau khi đã activate .venv, nâng cấp pip:

python -m pip install --upgrade pip

Cài theo requirements.txt:

python -m pip install --no-cache-dir -r requirements.txt

Hoặc cài đúng version đã freeze theo requirements-lock.txt:

python -m pip install --no-cache-dir -r requirements-lock.txt

Khuyến nghị cho các thành viên trong team:

python -m pip install --no-cache-dir -r requirements-lock.txt

---

## 7. Nội dung requirements.txt

File requirements.txt nên giữ dạng gọn như sau:

numpy
pandas
matplotlib
scikit-learn
tensorflow
opencv-python
pillow
streamlit
gradio
jupyter
notebook
ipykernel

---

## 8. Freeze version

Sau khi cài thư viện xong, freeze toàn bộ version hiện tại bằng lệnh:

python -m pip freeze > requirements-lock.txt

Mục đích:
- requirements.txt: danh sách thư viện chính, dễ đọc.
- requirements-lock.txt: bản khóa version đầy đủ để các member cài giống nhau.

---

## 9. Kiểm tra môi trường

Chạy:

python check_env.py

Kết quả mong muốn:

OK: numpy
OK: pandas
OK: matplotlib
OK: scikit-learn
OK: tensorflow
OK: opencv-python
OK: pillow
OK: streamlit
OK: gradio
OK: sklearn.metrics
OK: ImageDataGenerator
OK: image_dataset_from_directory

Kiểm tra nhanh:

python -c "import numpy,pandas,matplotlib,sklearn,tensorflow,cv2,PIL,streamlit,gradio; print('ALL MAIN LIBRARIES OK')"

Kiểm tra riêng sklearn.metrics:

python -c "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay; print('SKLEARN METRICS OK')"

Kiểm tra riêng Keras image pipeline:

python -c "from tensorflow.keras.preprocessing.image import ImageDataGenerator; from tensorflow.keras.utils import image_dataset_from_directory; print('KERAS IMAGE PIPELINE OK')"

---

## 10. Nội dung check_env.py

File check_env.py dùng để kiểm tra Python version, thư viện chính, sklearn metrics, TensorFlow/Keras image pipeline và CPU/GPU.

Nội dung check_env.py:

import sys

print("=" * 60)
print("PYTHON VERSION")
print("=" * 60)
print(sys.version)

packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("tensorflow", "tensorflow"),
    ("opencv-python", "cv2"),
    ("pillow", "PIL"),
    ("streamlit", "streamlit"),
    ("gradio", "gradio"),
]

print("\nMAIN LIBRARIES")
print("=" * 60)

for package_name, import_name in packages:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "version unknown")
        print(f"OK: {package_name} - {version}")
    except Exception as e:
        print(f"ERROR: {package_name} - {e}")

print("\nSKLEARN METRICS")
print("=" * 60)

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    print("OK: sklearn.metrics")
except Exception as e:
    print(f"ERROR: sklearn.metrics - {e}")

print("\nKERAS IMAGE PIPELINE")
print("=" * 60)

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import image_dataset_from_directory
    print("OK: ImageDataGenerator")
    print("OK: image_dataset_from_directory")
except Exception as e:
    print(f"ERROR: Keras image pipeline - {e}")

print("\nTENSORFLOW DEVICE")
print("=" * 60)

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print(f"ERROR: TensorFlow device check - {e}")

---

## 11. Lưu ý về TensorFlow GPU trên Windows

Nếu khi chạy check_env.py thấy:

TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.
GPU: []

Đây không phải lỗi.

Ý nghĩa:
- TensorFlow vẫn chạy bình thường bằng CPU.
- GPU không được dùng trên native Windows với TensorFlow >= 2.11.

Với đồ án sinh viên, train bằng CPU vẫn được. Nếu train quá chậm, có thể dùng:
- Google Colab
- WSL2
- TensorFlow-DirectML

---

## 12. Cấu hình interpreter trong VS Code

Trong VS Code:

Ctrl + Shift + P

Chọn:

Python: Select Interpreter

Chọn Python trong venv:

D:\School\traffic-sign-recognition\.venv\Scripts\python.exe

Sau khi chọn đúng, góc dưới VS Code nên hiện:

Python 3.11.x (.venv)

---

## 13. Chuẩn bị notebook cho Week 1

Tuần 1 cần chuẩn bị các notebook:

notebooks/01_data_exploration.ipynb
notebooks/02_cnn_baseline.ipynb
notebooks/03_transfer_learning.ipynb
notebooks/04_evaluation_demo.ipynb

Có thể tạo trực tiếp trong VS Code:

Right click notebooks/ → New File → 01_data_exploration.ipynb
Right click notebooks/ → New File → 02_cnn_baseline.ipynb
Right click notebooks/ → New File → 03_transfer_learning.ipynb
Right click notebooks/ → New File → 04_evaluation_demo.ipynb

---

## 14. Chuẩn bị folder dataset cho Week 1

Tạo cấu trúc:

data/processed/
├── train/
├── val/
└── test/

Nếu chưa có thì chạy:

mkdir data\processed\train
mkdir data\processed\val
mkdir data\processed\test

Khi dùng image_dataset_from_directory, mỗi class nên nằm trong một folder riêng.

Ví dụ:

data/processed/train/
├── class_0/
├── class_1/
├── class_2/
...

Hoặc:

data/processed/train/
├── 0/
├── 1/
├── 2/
...

Khi đó TensorFlow có thể đọc dataset bằng:

import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/train",
    image_size=(64, 64),
    batch_size=32
)

---

## 15. Hướng chuẩn bị Week 1

Week 1 tập trung vào:

1. Chọn dataset.
2. Tải dataset.
3. Khám phá dataset.
4. Kiểm tra số class.
5. Kiểm tra số ảnh mỗi class.
6. Chia dữ liệu train/val/test.
7. Test đọc ảnh bằng image_dataset_from_directory.
8. Tạo CNN baseline đầu tiên.
9. Tạo transfer learning baseline đầu tiên.

Dataset đề xuất:

GTSRB - German Traffic Sign Recognition Benchmark

Lý do:
- Phù hợp bài toán nhận diện biển báo giao thông.
- Có nhiều class biển báo.
- Dễ dùng cho CNN và transfer learning.
- Dễ đưa vào báo cáo/paper.

---

## 16. Các lệnh thường dùng

Kích hoạt venv:

.venv\Scripts\activate

Cài thư viện chính:

python -m pip install --no-cache-dir -r requirements.txt

Cài đúng version đã freeze:

python -m pip install --no-cache-dir -r requirements-lock.txt

Kiểm tra môi trường:

python check_env.py

Freeze version:

python -m pip freeze > requirements-lock.txt

Chạy Streamlit sau này:

streamlit run app/streamlit_app.py

Chạy Gradio sau này:

python app/gradio_app.py

---

## 17. Git notes

Không push các phần sau lên GitHub:

.venv/
data/raw/
data/processed/
models/*.keras
models/*.h5
__pycache__/
.ipynb_checkpoints/

Nên push:

README.md
requirements.txt
requirements-lock.txt
check_env.py
notebooks/
src/
app/

---

## 18. Checklist Week 0

[x] Cài Python 3.10/3.11
[x] Tạo virtual environment .venv
[x] Cài numpy, pandas, matplotlib
[x] Cài scikit-learn
[x] Cài tensorflow
[x] Cài opencv-python
[x] Cài pillow
[x] Cài streamlit hoặc gradio
[x] Cài jupyter, notebook, ipykernel
[x] Tạo requirements.txt
[x] Freeze requirements-lock.txt
[x] Tạo check_env.py
[x] Chạy python check_env.py
[x] Tạo cấu trúc thư mục project
[x] Tạo README.md
[x] Chọn interpreter .venv trong VS Code

---

## 19. Checklist trước khi sang Week 1

[ ] requirements.txt chỉ chứa danh sách thư viện chính
[ ] requirements-lock.txt đã được tạo bằng pip freeze
[ ] python check_env.py chạy OK
[ ] VS Code đã chọn đúng .venv
[ ] notebooks/01_data_exploration.ipynb đã tạo
[ ] notebooks/02_cnn_baseline.ipynb đã tạo
[ ] notebooks/03_transfer_learning.ipynb đã tạo
[ ] notebooks/04_evaluation_demo.ipynb đã tạo
[ ] data/processed/train đã có
[ ] data/processed/val đã có
[ ] data/processed/test đã có
[ ] Team thống nhất dùng dataset GTSRB hoặc dataset tương đương