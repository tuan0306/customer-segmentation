# Phân Cụm Khách Hàng (Customer Segmentation) với K-Means

## Giới thiệu Dự án
Dự án này thực hiện phân tích và phân khúc khách hàng cho một chuỗi bán lẻ trực tuyến tại Vương quốc Anh (UK) dựa trên dữ liệu giao dịch thực tế.

Mục tiêu là sử dụng thuật toán học máy không giám sát (**K-Means Clustering**) để chia cơ sở khách hàng thành các nhóm riêng biệt. Kết quả giúp doanh nghiệp tối ưu hóa chiến lược tiếp thị (Targeted Marketing) và giữ chân khách hàng (Customer Retention) thay vì tiếp thị đại trà.

---

## Cấu trúc Thư mục (Project Structure)
Dự án được tổ chức theo mô hình OOP (Lập trình hướng đối tượng) để code gọn gàng và dễ bảo trì.

```text
Customer Segmentation with KMeans/
├── data/
│   ├── raw/                    # Chứa dữ liệu gốc (online_retail.csv)
│   └── processed/              # Chứa dữ liệu đã làm sạch và file features
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb     # Bước 1: Làm sạch dữ liệu & EDA
│   ├── 02_feature_engineering.ipynb  # Bước 2: Tạo 16 đặc trưng & Box-Cox
│   └── 03_modeling.ipynb             # Bước 3: Phân cụm K-Means & Phân tích
├── src/
│   ├── __init__.py
│   └── clustering_library.py   # Thư viện chính chứa các class: DataCleaner, FeatureEngineer, ClusterAnalyzer
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn