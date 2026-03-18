import pandas as pd

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
