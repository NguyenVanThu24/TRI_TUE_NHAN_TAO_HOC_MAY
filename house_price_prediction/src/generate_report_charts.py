import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ==================== Paths ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")

# ==================== 1. Histogram (Before/After Log) ====================
plt.figure(figsize=(12, 5))

# Đồ thị 1: Biểu đồ giá nhà gốc
plt.subplot(1, 2, 1)
sns.histplot(train['SalePrice'], kde=True, color='blue', bins=50)
plt.title('Phân phối Giá nhà (Trước khi đổi Log)')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')

# Đồ thị 2: Biểu đồ sau khi Log
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train['SalePrice']), kde=True, color='green', bins=50)
plt.title('Phân phối Giá nhà (Sau khi đổi Log)')
plt.xlabel('Log(SalePrice)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(REPORT_DIR / '1_histogram_log_transform.png')
print("Saved: 1_histogram_log_transform.png")
plt.close()

# ==================== Feature Engineering ====================
train["TotalSF"] = train["TotalBsmtSF"].fillna(0) + train["1stFlrSF"].fillna(0) + train["2ndFlrSF"].fillna(0)
train["TotalBath"] = train["FullBath"].fillna(0) + 0.5 * train["HalfBath"].fillna(0) + train["BsmtFullBath"].fillna(0) + 0.5 * train["BsmtHalfBath"].fillna(0)

y = np.log1p(train["SalePrice"])
X = train.drop(["SalePrice", "Id"], axis=1)

# ==================== Preprocessing ====================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split Test/Train to avoid overfitting metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== Models ====================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
}

# ==================== Evaluate Models (Table 4.4.1) ====================
print("\n" + "="*60)
print(f"{'Mô hình':<25} | {'R² (Train)':<10} | {'R² (Test)':<10} | {'RMSLE':<10}")
print("="*60)

best_model_name = ""
best_model_preds = None

for name, model in models.items():
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    
    # Huấn luyện
    clf.fit(X_train, y_train)
    
    # Dự đoán
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    
    # Tính điểm
    r2_train = r2_score(y_train, train_preds)
    r2_test = r2_score(y_test, test_preds)
    rmsle_test = np.sqrt(mean_squared_error(y_test, test_preds)) # Vì lúc đầu y đã log1p rồi, nên tính RMSE trên y_test tức là RMSLE
    
    print(f"{name:<25} | {r2_train:<10.3f} | {r2_test:<10.3f} | {rmsle_test:<10.3f}")
    
    if name == "XGBoost Regressor":
        best_model_preds = test_preds

print("="*60 + "\n")

# ==================== 2. Scatter Plot (Actual vs Predicted) ====================
plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_model_preds, alpha=0.5, color='orange')
# Đường chéo lý tưởng
p1 = max(max(best_model_preds), max(y_test))
p2 = min(min(best_model_preds), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b--', lw=2)

plt.xlabel('Giá trị Thực tế (Actual Log Price)', fontsize=12)
plt.ylabel('Giá trị Dự đoán (Predicted Log Price)', fontsize=12)
plt.title('So sánh Giá dự đoán và Thực tế (Mô hình XGBoost)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(REPORT_DIR / '2_scatter_actual_vs_predicted.png')
print("Saved: 2_scatter_actual_vs_predicted.png (Scatter plot)")

print("\nHoàn tất! Các biểu đồ đã được lưu trong thư mục 'reports/figures/'")
