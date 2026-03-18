# <p align="center">TRÍ TUỆ NHÂN TẠO & HỌC MÁY</p>

**Họ và tên:** Nguyễn Văn Thứ

**Lớp:** K58KTP

**MSSV:** K225480106062

**Giáo viên hướng dẫn:** TS. Nguyễn Tuấn Linh 

**Bài tập lớn môn:** Trí tuệ nhân tọa & học máy

**Đề tài:** Xây dựng hệ thống dự đoán giá nhà dựa trên các đặc trưng bất động sản sử dụng học máy.

**Link video Youtube báo cáo và demo kết quả:** https://youtu.be/E6c_JKXwlRc

## Hướng dẫn cài đặt và chạy chương trình
#### 1. Yêu cầu hệ thống   
- Hệ điều hành: Windows / Linux / macOS   
- Python phiên bản 3.8 trở lên   
- Kết nối Internet để cài thư viện
#### 2. Các công cụ sử dụng
- Ngôn ngữ lập trình: Python   
- Thư viện chính:   
  - Streamlit: xây dựng giao diện web   
  - Pandas: xử lý dữ liệu   
  - NumPy: tính toán số học   
  - XGBoost: xây dựng mô hình dự đoán   
  - Scikit-learn: hỗ trợ tiền xử lý và đánh giá mô hình   
- Công cụ phát triển:   
  - Visual Studio Code / PyCharm   
  - GitHub (quản lý mã nguồn)
#### 3. Hướng dẫn cài đặt   
**Bước 1:** Cài đặt Python   
- Tải và cài đặt Python tại trang chính thức: https://www.python.org   
- Kiểm tra sau khi cài: `python --version`

**Bước 2:** Tải mã nguồn   
- Clone từ GitHub: `git clone <link-github-cua-ban>` Hoặc tải file .zip và giải nén.

**Bước 3:** Cài đặt thư viện   
- Di chuyển vào thư mục dự án: `cd house_price_prediction`   
- Cài đặt các thư viện cần thiết: `pip install -r requirements.txt`
#### 4. Hướng dẫn chạy chương trình
- Chạy ứng dụng bằng lệnh: `streamlit run app.py`   
- Sau khi chạy thành công, hệ thống sẽ tự động mở trình duyệt tại địa chỉ: `http://localhost:8501`
#### 5. Hướng dẫn sử dụng
- Nhập các thông tin của căn nhà: Diện tích, Số phòng ngủ, phòng tắm, Năm xây dựng, Vị trí
- Hệ thống sẽ tự động:
  - Xử lý dữ liệu đầu vào
  - Đưa vào mô hình dự đoán
  - Hiển thị giá nhà dự đoán
#### 6. Ghi chú
- Đảm bảo đã cài đầy đủ thư viện trước khi chạy
- Nếu lỗi thiếu thư viện, có thể cài thủ công: `pip install streamlit pandas numpy scikit-learn xgboost`
- Trong trường hợp cổng 8501 bị chiếm dụng, có thể đổi cổng khác: `streamlit run app.py --server.port 8502`

## Link dữ liệu, link code trên github
#### 1. Link dữ liệu: 
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
#### 2. Link và Mã QR Github:

<img width="431" height="431" alt="Ảnh chụp màn hình 2026-03-18 231722" src="https://github.com/user-attachments/assets/ca8e146a-6de0-421b-8e72-2834d2a7ff94" />
