import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos = (y_train.value_counts()[0] / y_train.value_counts()[1])

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos
)

xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, "model/xgb_model.pkl")
joblib.dump(X.columns, "model/columns.pkl")

print("✅ Model and columns saved successfully!")