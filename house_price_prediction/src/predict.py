import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
SUBMISSION_DIR = BASE_DIR / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

print("Loading trained models...")

ridge = joblib.load(MODEL_DIR / "ridge.pkl")
gbr = joblib.load(MODEL_DIR / "gbr.pkl")
xgb = joblib.load(MODEL_DIR / "xgb.pkl")

print("Loading test data...")
test = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test["Id"]

# =============================
# Feature Engineering (MUST SAME AS TRAIN)
# =============================
print("Creating features...")

test["TotalSF"] = (
    test["TotalBsmtSF"].fillna(0)
    + test["1stFlrSF"].fillna(0)
    + test["2ndFlrSF"].fillna(0)
)

test["TotalBath"] = (
    test["FullBath"].fillna(0)
    + 0.5 * test["HalfBath"].fillna(0)
    + test["BsmtFullBath"].fillna(0)
    + 0.5 * test["BsmtHalfBath"].fillna(0)
)

test = test.drop(["Id"], axis=1)

# =============================
# Predict
# =============================
print("Predicting...")

pred1 = ridge.predict(test)
pred2 = gbr.predict(test)
pred3 = xgb.predict(test)

# Ensemble (Ridge đang tốt nên weight cao hơn)
# Bỏ Ridge
preds_log = 0.3 * pred2 + 0.7 * pred3


# Back to normal scale
preds = np.expm1(preds_log)

# Fix inf / nan nếu có
preds = np.nan_to_num(preds, nan=np.median(preds))

# Cap extreme values nhẹ
upper_cap = np.percentile(preds, 99.5)
preds = np.clip(preds, None, upper_cap)

# =============================
# Submission
# =============================
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": preds
})

submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

print("Submission file created successfully!")
