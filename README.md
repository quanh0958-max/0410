# 📈 Nền Tảng Kiểm Định Chiến Lược Giao Dịch Định Lượng HOSE (SMA + OBV + MPT)

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit**, cho phép kiểm định (Backtesting) tính hiệu quả của chiến lược kết hợp chỉ báo xu hướng **SMA (Simple Moving Average)** và chỉ báo dòng tiền **OBV (On-Balance Volume)**, đồng thời so sánh hai mô hình phân bổ danh mục đầu tư: **Equal Weight (1/N)** và **Modern Portfolio Theory (MPT - Markowitz Max Sharpe)** trên dữ liệu cổ phiếu niêm yết tại Sở Giao dịch Chứng khoán TP.HCM (HOSE).

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Bộ lọc dữ liệu & Tùy biến cổ phiếu linh hoạt**:
   - Tự động nhận diện toàn bộ các mã cổ phiếu có trong tệp dữ liệu sàn HOSE (hơn 100 mã).
   - Mặc định lựa chọn 3 cổ phiếu đại diện 3 nhóm ngành trọng yếu: **ACB** (Ngân hàng), **FPT** (Công nghệ), **HPG** (Sản xuất công nghiệp).
   - Tự do thay đổi số lượng và danh sách mã cổ phiếu trong danh mục.
2. **Quy trình kiểm định chuẩn hóa (No Look-Ahead Bias)**:
   - Phân chia chuỗi thời gian nghiêm ngặt: **Tập Train (2020 - 2021)** dùng để tối ưu tham số và ước lượng trọng số; **Tập Test (2022)** ngoài mẫu (Out-of-sample) dùng để đánh giá độ bền vững.
   - Tín hiệu giao dịch áp dụng cơ chế **dịch 1 phiên (Shift 1)**: Tín hiệu sinh ra từ giá đóng cửa phiên $t$ chỉ thực thi tại phiên $t+1$.
3. **Tối ưu hóa tham số**:
   - Tích hợp thuật toán tối ưu xác suất **Hyperopt (TPE - Tree-structured Parzen Estimator)** để tìm bộ tham số $(ma_{short}, ma_{long}, obv_{window})$ tối đa hóa Sharpe Ratio trên tập Train.
   - Cung cấp sẵn bộ tham số tối ưu chuẩn cho tốc độ tính toán tức thì, kèm tùy chọn tinh chỉnh thủ công.
4. **Mô hình phân bổ danh mục hiện đại (Markowitz MPT vs Equal Weight)**:
   - Giải bài toán tối ưu lồi (SLSQP) tìm véc-tơ trọng số tối đa hóa Sharpe Ratio trên Train với ràng buộc $0 \le w_i \le 1$ và $\sum w_i = 1$.
   - Giữ nguyên bộ trọng số đã xác định để kiểm thử trực tiếp trên dữ liệu Test 2022.
5. **Trực quan hóa tài chính chuyên sâu với Plotly**:
   - Biểu đồ tăng trưởng vốn danh mục (**Portfolio Equity Curve**) và biểu đồ sụt giảm tối đa (**Underwater Drawdown**).
   - Biểu đồ kỹ thuật đa tầng (Multi-panel): Giá, SMA, tín hiệu Mua/Bán (Tam giác xanh/đỏ) và dao động khối lượng OBV kèm OBV Moving Average.
   - Bảng so sánh hiệu suất toàn diện: *Total Return, CAGR, Annual Volatility, Sharpe Ratio, Max Drawdown, Win Rate*.
   - Cho phép xuất báo cáo thống kê định dạng CSV.

---

## 📐 Phương Pháp Luận Định Lượng

### 1. Chỉ báo kỹ thuật
- **SMA Crossover**:
  - Tín hiệu **MUA (Buy = 1)**: Khi $SMA_{short}$ cắt lên trên $SMA_{long}$ ($SMA_{short, t} > SMA_{long, t}$ và $SMA_{short, t-1} \le SMA_{long, t-1}$).
  - Tín hiệu **BÁN (Sell = -1)**: Khi $SMA_{short}$ cắt xuống dưới $SMA_{long}$ ($SMA_{short, t} < SMA_{long, t}$ và $SMA_{short, t-1} \ge SMA_{long, t-1}$).
- **OBV Breakout**:
  - $OBV_t = OBV_{t-1} + \text{sign}(Close_t - Close_{t-1}) \times Volume_t$.
  - Đường trung bình động khối lượng: $OBV\_MA = \text{SMA}(OBV, \text{window})$.
  - Tín hiệu **MUA (Buy = 1)**: Khi $OBV$ cắt lên trên $OBV\_MA$.
  - Tín hiệu **BÁN (Sell = -1)**: Khi $OBV$ cắt xuống dưới $OBV\_MA$.

### 2. Logic kết hợp tín hiệu
- **Chế độ OR**: Tận dụng cơ hội sớm khi ít nhất một trong hai chỉ báo phát tín hiệu. Trường hợp xuất hiện mâu thuẫn đồng thời (một chỉ báo báo Mua, chỉ báo kia báo Bán), hệ thống giữ vị thế trung lập (0).
- **Chế độ AND**: Chiến lược phòng vệ cao, chỉ vào lệnh khi cả hai chỉ báo cùng đồng thuận xác nhận xu hướng và dòng tiền.

### 3. Tối ưu danh mục Modern Portfolio Theory (MPT)
- Tối đa hóa tỷ số Sharpe trên tập Train:
  $$\max_{w} \quad \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
  với $\mu$ là véc-tơ lợi nhuận trung bình năm, $\Sigma$ là ma trận hiệp phương sai năm của các chiến lược thành phần, $R_f$ là lãi suất phi rủi ro.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                              # Mã nguồn ứng dụng Streamlit Web App chính
├── requirements.txt                    # Danh sách thư viện Python cần thiết
├── README.md                           # Tài liệu giới thiệu, hướng dẫn sử dụng & triển khai
├── HOSE_2020_2023_in.csv               # Tệp dữ liệu lịch sử giá sàn HOSE (2020 - 2022)
└── HOSE_SMA_OBV_EqualWeight_MPT.ipynb  # Notebook phân tích định lượng gốc
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Đã cài đặt **Python** (phiên bản 3.9, 3.10 hoặc 3.11).
- Trình quản lý gói `pip` và công cụ `git`.

### 2. Các bước khởi chạy
```bash
# 1. Di chuyển vào thư mục dự án
cd "duong_dan_den_thu_muc"

# 2. Tạo môi trường ảo (Khuyên dùng)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
.\venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Chạy ứng dụng Streamlit
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Triển Khai Lên Streamlit Community Cloud (Miễn Phí)

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên: `hose-quant-backtest`).
2. Mở terminal tại thư mục chứa các file (`app.py`, `requirements.txt`, `README.md`, `HOSE_2020_2023_in.csv`) và thực hiện:
   ```bash
   git init
   git add .
   git commit -m "Khoi tao du an Streamlit HOSE Quant Backtest"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/hose-quant-backtest.git
   git push -u origin main
   ```
   *(Lưu ý: Tệp `HOSE_2020_2023_in.csv` có dung lượng ~5.7 MB, hoàn toàn nằm trong giới hạn cho phép của GitHub mà không cần cài đặt Git LFS).*

### Bước 2: Triển khai trên Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản **GitHub**.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin:
   - **Repository**: Chọn repository bạn vừa đẩy lên (ví dụ: `<tai-khoan-github>/hose-quant-backtest`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - *(Tùy chọn)* **App URL**: Bạn có thể tùy chỉnh đường dẫn URL theo ý muốn (ví dụ: `hose-quant-backtest.streamlit.app`).
4. Nhấn **"Deploy!"**.
5. Hệ thống máy chủ Streamlit sẽ tự động đọc `requirements.txt`, cài đặt môi trường và khởi chạy ứng dụng sau khoảng 1 - 2 phút.

---

## 📊 Kết Quả Thực Nghiệm & Phân Tích Định Lượng

Dựa trên dữ liệu thực tế sàn HOSE qua 2 giai đoạn đối lập:
1. **Giai đoạn Train (2020 - 2021) - Chu kỳ tăng giá (Bull Market)**:
   - Các cổ phiếu đầu ngành như HPG, FPT, ACB tăng trưởng vượt bậc.
   - Danh mục MPT đạt lợi nhuận lũy kế **152.56%** (Sharpe 2.19), nhỉnh hơn danh mục Equal Weight đạt **138.32%** (Sharpe 2.15) nhờ MPT dồn tỷ trọng cao vào các mã có đà tăng mạnh nhất (HPG ~41.8%, ACB ~40.8%).
2. **Giai đoạn Test (2022) - Chu kỳ điều chỉnh sâu (Bear Market)**:
   - Thị trường đối mặt với đợt bán tháo trên diện rộng; hầu hết cổ phiếu VN-Index sụt giảm mạnh.
   - **Hiện tượng Overfitting & Estimation Risk của MPT**: Do MPT tối ưu hóa dựa hoàn toàn vào dữ liệu chu kỳ tăng trưởng quá khứ, khi bước vào thị trường giá xuống, danh mục MPT chịu sụt giảm sâu hơn (**-18.76%**, Max Drawdown **-21.53%**) so với danh mục phân bổ đều Equal Weight (**-17.65%**, Max Drawdown **-20.02%**).
   - **Bài học định lượng**: Mô hình phân bổ đều **Equal Weight (1/N)** cho thấy khả năng chống chịu rủi ro và phòng vệ tốt hơn trước sự đảo chiều đột ngột của chế độ thị trường (Regime Shift).

---

## 👥 Tác Giả & Bản Quyền
Dự án được xây dựng phục vụ nghiên cứu học thuật và kiểm định chiến lược đầu tư tài chính định lượng trên thị trường chứng khoán Việt Nam.
