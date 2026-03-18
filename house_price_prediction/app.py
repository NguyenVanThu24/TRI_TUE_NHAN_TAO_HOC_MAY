import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
# streamlit run app.py lệnh chạy

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

# =============================
# Load model
# =============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =============================
# Load sample data
# =============================
df_sample = pd.read_csv(DATA_PATH)

# =============================
# Neighborhood Mapping (Kaggle -> VN)
# =============================
vn_neighborhoods = {
    'CollgCr': 'Quận 1, TP.HCM', 'Veenker': 'Quận 3, TP.HCM',
    'Crawfor': 'Quận Phú Nhuận, TP.HCM', 'NoRidg': 'Quận 2 (Thảo Điền), TP.HCM',
    'Mitchel': 'Quận Bình Thạnh, TP.HCM', 'Somerst': 'Quận 7 (Phú Mỹ Hưng), TP.HCM',
    'NWAmes': 'Quận Tân Bình, TP.HCM', 'OldTown': 'Quận Hoàn Kiếm, Hà Nội',
    'BrkSide': 'Quận Đống Đa, Hà Nội', 'Sawyer': 'Quận Cầu Giấy, Hà Nội',
    'NridgHt': 'Quận Tây Hồ, Hà Nội', 'NAmes': 'Quận Thanh Xuân, Hà Nội',
    'SawyerW': 'Quận Hai Bà Trưng, Hà Nội', 'IDOTRR': 'Quận Nam Từ Liêm, Hà Nội',
    'MeadowV': 'Quận Hoàng Mai, Hà Nội', 'Edwards': 'Quận Long Biên, Hà Nội',
    'Timber': 'Quận Sơn Trà, Đà Nẵng', 'Gilbert': 'Quận Hải Châu, Đà Nẵng',
    'StoneBr': 'Quận Ngũ Hành Sơn, Đà Nẵng', 'ClearCr': 'Quận Thanh Khê, Đà Nẵng',
    'NPkVill': 'TP. Nha Trang, Khánh Hòa', 'Blmngtn': 'TP. Đà Lạt, Lâm Đồng',
    'BrDale': 'TP. Vũng Tàu, BR-VT', 'SWISU': 'TP. Cần Thơ', 'Blueste': 'TP. Phú Quốc, Kiên Giang'
}
# Hệ số đắt đỏ của Bất Động Sản Việt Nam so với mặt bằng Kaggle (Giả lập)
# Đã được điều chỉnh giảm để sát giá thực tế (Ví dụ: Nhà 100m2 Quận 1 hiện tại ~ 15-25 Tỷ)
location_multipliers = {
    'Quận 1, TP.HCM': 5.0, 'Quận 3, TP.HCM': 4.0, 
    'Quận Phú Nhuận, TP.HCM': 3.5, 'Quận 2 (Thảo Điền), TP.HCM': 4.5,
    'Quận Bình Thạnh, TP.HCM': 3.0, 'Quận 7 (Phú Mỹ Hưng), TP.HCM': 4.0,
    'Quận Tân Bình, TP.HCM': 2.5, 'Quận Hoàn Kiếm, Hà Nội': 6.0,
    'Quận Đống Đa, Hà Nội': 4.5, 'Quận Cầu Giấy, Hà Nội': 4.0,
    'Quận Tây Hồ, Hà Nội': 5.5, 'Quận Thanh Xuân, Hà Nội': 3.5,
    'Quận Hai Bà Trưng, Hà Nội': 4.2, 'Quận Nam Từ Liêm, Hà Nội': 2.5,
    'Quận Hoàng Mai, Hà Nội': 2.0, 'Quận Long Biên, Hà Nội': 2.5,
    'Quận Sơn Trà, Đà Nẵng': 2.0, 'Quận Hải Châu, Đà Nẵng': 2.5,
    'Quận Ngũ Hành Sơn, Đà Nẵng': 1.8, 'Quận Thanh Khê, Đà Nẵng': 1.5,
    'TP. Nha Trang, Khánh Hòa': 1.8, 'TP. Đà Lạt, Lâm Đồng': 2.0,
    'TP. Vũng Tàu, BR-VT': 1.5, 'TP. Cần Thơ': 1.0, 'TP. Phú Quốc, Kiên Giang': 2.2
}
# Đảo ngược mapping để từ Tiếng Việt suy ra lại chuẩn Kaggle đưa vào Model
reverse_mapping = {v: k for k, v in vn_neighborhoods.items()}


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# =============================
# Sidebar Info
# =============================
st.sidebar.title("📊 Thông tin Mô hình")
st.sidebar.info(
    "Mô hình sử dụng: **XGBoost Regressor**\n\n"
    "Độ chính xác CV (R² Score): **~89.6%** \n"
    "*(Điểm đã qua K-Fold Cross Validation để tránh Overfitting)*\n\n"
    "Sai số RMSLE (Log Scale): **~0.125** \n"
    "*(Sai số trên thang đo Logarit, tương đương lệch ~12.5% giá trị)*"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Tính năng mới")
st.sidebar.markdown(
    "- Trực quan hóa mức độ đắt/rẻ của căn nhà so với toàn bộ thị trường.\n"
    "- Biểu đồ **Histogram** sẽ tự động được vẽ dựa trên kết quả cuối cùng."
)

st.title("🏠 Dự đoán giá nhà")
st.write("Demo mô hình Machine Learning dự đoán giá nhà")

# =============================
# Input form
# =============================
with st.form("house_form"):
    st.subheader("Nhập thông số căn nhà")

    col1, col2 = st.columns(2)
    with col1:
        OverallQual = st.slider("Chất lượng hiện tại (1–10)", 1, 10, 5)
        GrLivArea_m2 = st.number_input("Diện tích sinh hoạt (m²)", 30, 1000, 100)
        TotalBsmtSF_m2 = st.number_input("Diện tích tầng hầm (m²)", 0, 500, 0)
        YearBuilt = st.slider("Năm xây dựng", 1950, 2024, 2010)
        
        # Thêm loại đường (Mặt tiền / Hẻm)
        street_type = st.radio("Vị trí đường", ["Mặt tiền đường lớn", "Trong hẻm nhỏ"])
        
    with col2:
        BedroomAbvGr = st.slider("Số phòng ngủ", 1, 10, 3)
        FullBath = st.slider("Số phòng tắm đầy đủ", 0, 6, 2)
        GarageCars = st.slider("Sức chứa Gara (số xe hơi)", 0, 6, 0)
        
        # Thêm Pháp lý (Sổ đỏ / Đang tranh chấp)
        sale_condition = st.selectbox("Tình trạng pháp lý", ["Sổ đỏ chính chủ (Bình thường)", "Đang thế chấp/Thanh lý", "Nhà mới xây chưa hoàn công"])
        
        # Chỉ lấy những khu vực có map trong dict
        valid_areas = [vn_neighborhoods[str(n)] for n in df_sample["Neighborhood"].unique() if str(n) in vn_neighborhoods]
        vn_neighborhood_select = st.selectbox(
            "Khu vực (vị trí nhà)",
            valid_areas
        )

    st.markdown("---")
    submit = st.form_submit_button("🔮 Dự đoán giá ngay", use_container_width=True)

# =============================
# Prediction
# =============================
if submit:
    # Lấy lại tên chuẩn Kaggle từ Tiếng Việt
    kaggle_neighborhood = reverse_mapping[vn_neighborhood_select]
    
    # Map loại đường sang chuẩn Kaggle (Street/Alley)
    # Pave: Mặt đường rải nhựa, Grvl: Đường rải sỏi (Hẻm)
    kaggle_street = "Pave" if street_type == "Mặt tiền đường lớn" else "Grvl"
    
    # Map Pháp lý sang chuẩn Kaggle (SaleCondition)
    # Normal: Sổ đỏ, Abnorml: Thế chấp/Tranh chấp, Partial: Nhà mới
    if sale_condition == "Sổ đỏ chính chủ (Bình thường)":
        kaggle_sale_cond = "Normal"
    elif sale_condition == "Nhà mới xây chưa hoàn công":
        kaggle_sale_cond = "Partial"
    else:
        kaggle_sale_cond = "Abnorml"

    input_data = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea_m2 * 10.7639,  # Chuyển đổi m2 sang sq.ft
        "TotalBsmtSF": TotalBsmtSF_m2 * 10.7639, # Chuyển đổi m2 sang sq.ft
        "GarageCars": GarageCars,
        "YearBuilt": YearBuilt,
        "FullBath": FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "Street": kaggle_street,
        "SaleCondition": kaggle_sale_cond,
        "Neighborhood": kaggle_neighborhood
    }

    input_df = pd.DataFrame([input_data])

    # Các feature còn thiếu → lấy median / mode từ train
    for col in df_sample.columns:
        if col not in input_df.columns and col not in ["SalePrice", "Id"]:
            if df_sample[col].dtype == "object":
                input_df[col] = df_sample[col].mode()[0]
            else:
                input_df[col] = df_sample[col].median()

    # Feature engineering
    input_df["TotalSF"] = (
        input_df["TotalBsmtSF"]
        + input_df["1stFlrSF"]
        + input_df["2ndFlrSF"]
    )

    input_df["TotalBath"] = (
        input_df["FullBath"]
        + 0.5 * input_df["HalfBath"]
        + input_df["BsmtFullBath"]
        + 0.5 * input_df["BsmtHalfBath"]
    )

    pred_log = model.predict(input_df)[0]
    pred_price_usd = np.expm1(pred_log)
    
    # Ứng dụng "Hệ số Giá Bất Động Sản Việt Nam"
    # Công thức: Tỷ giá (25,000) * Hệ số Khu vực (VD: Quận 1 x 10.5) 
    multiplier = location_multipliers.get(vn_neighborhood_select, 3.0) 
    pred_price_vnd = pred_price_usd * 25000 * multiplier

    if pred_price_vnd >= 1e9:
        price_str = f"{pred_price_vnd / 1e9:,.2f} Tỷ VNĐ"
    else:
        price_str = f"{pred_price_vnd / 1e6:,.0f} Triệu VNĐ"

    st.success(f"💰 Giá dự đoán: **{price_str}**")

    # -------------------------
    # Chart: Compare to market
    # -------------------------
    import matplotlib.pyplot as plt

    st.markdown("---")
    st.subheader(f"📈 Phân tích Mức giá tại {vn_neighborhood_select}")
    st.write("Vị trí về giá của căn nhà so với phân khúc khu vực này:")

    # Lấy ra các căn nhà tương đương trong tập Kaggle và gắn hệ số của khu vực hiện tại
    # Để đồ thị hiển thị đúng phân khúc giá của khu vực mà người dùng vừa chọn.
    df_price_billion_vnd = (df_sample["SalePrice"].dropna() * 25000 * multiplier) / 1e9
    pred_price_billion = pred_price_vnd / 1e9

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_price_billion_vnd, bins=50, color="#6eb5ff", edgecolor="white", alpha=0.8)
    
    # Kẻ đường đứt khúc hiển thị giá dự đoán
    ax.axvline(pred_price_billion, color="#ff4c4c", linestyle="dashed", linewidth=2.5)
    
    ylim_max = ax.get_ylim()[1]
    ax.text(
        pred_price_billion * 1.05, 
        ylim_max * 0.85, 
        f"Giá dự đoán:\n{pred_price_billion:,.2f} Tỷ", 
        color="#ff4c4c", 
        fontweight="bold"
    )

    ax.set_xlabel("Mức giá (Tỷ VNĐ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Số lượng nhà", fontsize=11, fontweight="bold")
    ax.set_title("Biểu đồ Phân phối Mức Giá Bất Động Sản", fontsize=13, fontweight="bold")

    # Xóa viền trên và phải
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)
