import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Canh bao lech lac du lieu (Data Drift)
    pos_ratio = float(y_train.mean())
    if abs(pos_ratio - 0.248) > 0.05:
        print(f"CANH BAO (Data Drift): Ty le lop duong ({pos_ratio:.4f}) lech qua 5% so voi moc 24.8%!")
    else:
        print(f"Kiem tra phan phoi: Ty le lop duong = {pos_ratio:.4f} (on dinh)")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Dieu chinh nguong quyet dinh (Threshold Tuning)
        probs = model.predict_proba(X_eval)[:, 1]
        best_thresh = 0.5
        best_f1 = f1
        for th in np.arange(0.1, 0.95, 0.05):
            th_preds = (probs >= th).astype(int)
            th_f1 = float(f1_score(y_eval, th_preds))
            if th_f1 > best_f1:
                best_f1 = th_f1
                best_thresh = round(float(th), 2)

        print(f"Threshold Tuning: Nguong toi uu = {best_thresh} voi F1 = {best_f1:.4f} (nguong mac dinh 0.5: F1 = {f1:.4f})")

        # Bonus 3: Tao bao cao Precision / Recall tu dong
        cm = confusion_matrix(y_eval, preds)
        cls_report = classification_report(y_eval, preds, target_names=["<=50K", ">50K"])
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cls_report)

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("=== CONFUSION MATRIX ===\n")
            f.write(str(cm) + "\n\n")
            f.write("=== CLASSIFICATION REPORT ===\n")
            f.write(cls_report)

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_ratio", pos_ratio)
        mlflow.log_metric("best_threshold", best_thresh)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 8: Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "positive_ratio": pos_ratio,
                "best_threshold": best_thresh,
                "best_f1_score": best_f1,
            }, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
