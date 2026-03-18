import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")

# =============================
# Feature Engineering
# =============================
print("Creating new features...")

train["TotalSF"] = (
    train["TotalBsmtSF"].fillna(0)
    + train["1stFlrSF"].fillna(0)
    + train["2ndFlrSF"].fillna(0)
)

train["TotalBath"] = (
    train["FullBath"].fillna(0)
    + 0.5 * train["HalfBath"].fillna(0)
    + train["BsmtFullBath"].fillna(0)
    + 0.5 * train["BsmtHalfBath"].fillna(0)
)

# =============================
# Target
# =============================
y = np.log1p(train["SalePrice"])
X = train.drop(["SalePrice", "Id"], axis=1)

# =============================
# Preprocessing
# =============================
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

# =============================
# Models
# =============================
ridge = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=20))
])

gbr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

# =============================
# Cross Validation
# =============================
print("Evaluating models...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(
        -cross_val_score(
            model,
            X,
            y,
            scoring="neg_mean_squared_error",
            cv=kf
        )
    )
    return rmse.mean()

print("Ridge RMSE:", rmse_cv(ridge))
print("GradientBoosting RMSE:", rmse_cv(gbr))
print("XGBoost RMSE:", rmse_cv(xgb))

# =============================
# Train full models
# =============================
print("Training full models...")

ridge.fit(X, y)
gbr.fit(X, y)
xgb.fit(X, y)

joblib.dump(ridge, MODEL_DIR / "ridge.pkl")
joblib.dump(gbr, MODEL_DIR / "gbr.pkl")
joblib.dump(xgb, MODEL_DIR / "xgb.pkl")

print("All models saved successfully!")

import matplotlib.pyplot as plt

print("Generating feature importance...")

# Lấy model XGBoost đã train (pipeline)
xgb_model = xgb.named_steps["model"]

# Lấy tên feature sau preprocessing (đã one-hot)
feature_names = xgb.named_steps["preprocessor"].get_feature_names_out()

importances = xgb_model.feature_importances_

# Tạo DataFrame
feat_imp = pd.Series(importances, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False).head(20)

# Plot
plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png")
plt.show()
