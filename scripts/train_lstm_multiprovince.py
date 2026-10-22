"""
เทรน Hybrid LSTM (สาขา temporal รับ lag 5 ค่า: 24,12,6,3,1 ชม. ย้อนหลัง ผ่าน LSTM + สาขา static
รับ feature ที่เหลือผ่าน Dense แล้วรวมกัน) เปรียบเทียบกับ LightGBM/XGBoost/Persistence/Dynamic Routing

ใช้ data/split/feature ชุดเดียวกับ train_multiprovince_models.py วัด MAE/RMSE/R2 ต่อ horizon
(24 horizon แยกโมเดล, target เป็น residual) บน test set
"""
import copy
import importlib.util
import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = dt.now().strftime("%Y%m%d_%H%M%S")

SEED = 42
BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 5
LR = 1e-3
HIDDEN = 32

# sanity-check ตอนจบ: MAE นอกช่วงนี้ = น่าสงสัย ต้องตรวจก่อนรายงานผล ไม่ใช่เอาไปใส่เล่มทันที
MAE_SANITY_LOW = 2.70   # ดีกว่า LightGBM/XGBoost ที่จูนมาอย่างหนัก ไม่น่าเชื่อถ้าเกิดขึ้นจริง
MAE_SANITY_HIGH = 3.50  # แย่กว่า Persistence baseline ธรรมดา


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse DATA_PATH/FEATURE_COLS/split_train_calib_test ของจริง ไม่ copy — กัน split/feature เพี้ยนจากต้นฉบับ
train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
FEATURE_COLS = train_mod.FEATURE_COLS

LAG_FEATURES = ["pm25_lag_24", "pm25_lag_12", "pm25_lag_6", "pm25_lag_3", "pm25_lag_1"]
STATIC_FEATURES = [c for c in FEATURE_COLS if c not in LAG_FEATURES]
assert set(LAG_FEATURES) | set(STATIC_FEATURES) == set(FEATURE_COLS)
assert len(LAG_FEATURES) == 5 and len(STATIC_FEATURES) == 15


class HybridLSTM(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.static_branch = nn.Sequential(nn.Linear(len(STATIC_FEATURES), hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, lag_seq, static_feat):
        _, (h_n, _) = self.lstm(lag_seq)
        lstm_out = h_n[-1]  # (batch, hidden)
        static_out = self.static_branch(static_feat)
        combined = torch.cat([lstm_out, static_out], dim=1)
        return self.head(combined).squeeze(-1)


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.randn(4, 5, 1, device="cuda", requires_grad=True)
        out, _ = nn.LSTM(1, 8, batch_first=True).to("cuda")(x)
        out.sum().backward()
        print(f"GPU พร้อมใช้จริง: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    except Exception as e:
        print(f"GPU report ว่าพร้อมแต่รันจริงไม่ได้ ({type(e).__name__}: {e}) — fallback CPU")
        return torch.device("cpu")


DEVICE = pick_device()


def to_tensors(df_scaled, feature_cols):
    lag_arr = df_scaled[LAG_FEATURES].values.astype(np.float32).reshape(-1, 5, 1)
    static_arr = df_scaled[STATIC_FEATURES].values.astype(np.float32)
    return torch.tensor(lag_arr), torch.tensor(static_arr)


def train_one_horizon(h, train_df, calib_df, test_df):
    target_col = f"pm25_t+{h}"
    # LightGBM/XGBoost จัดการ NaN ในฟีเจอร์ได้เอง แต่ neural net ทำไม่ได้ (ไหลผ่าน StandardScaler
    # เข้า loss กลายเป็น NaN ทั้งก้อน) จึง dropna ฟีเจอร์ด้วย
    dropna_cols = [target_col] + FEATURE_COLS
    train_h = train_df.dropna(subset=dropna_cols)
    calib_h = calib_df.dropna(subset=dropna_cols)
    test_h = test_df.dropna(subset=dropna_cols)

    scaler = StandardScaler().fit(train_h[FEATURE_COLS])  # fit บน train เท่านั้น
    train_scaled = pd.DataFrame(scaler.transform(train_h[FEATURE_COLS]), columns=FEATURE_COLS)
    calib_scaled = pd.DataFrame(scaler.transform(calib_h[FEATURE_COLS]), columns=FEATURE_COLS)
    test_scaled = pd.DataFrame(scaler.transform(test_h[FEATURE_COLS]), columns=FEATURE_COLS)

    y_tr_resid = (train_h[target_col].values - train_h["pm25"].values).astype(np.float32)
    y_calib_resid = (calib_h[target_col].values - calib_h["pm25"].values).astype(np.float32)

    lag_tr, static_tr = to_tensors(train_scaled, FEATURE_COLS)
    lag_calib, static_calib = to_tensors(calib_scaled, FEATURE_COLS)
    lag_test, static_test = to_tensors(test_scaled, FEATURE_COLS)

    torch.manual_seed(SEED)
    model = HybridLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(lag_tr, static_tr, torch.tensor(y_tr_resid))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    lag_calib_dev, static_calib_dev = lag_calib.to(DEVICE), static_calib.to(DEVICE)
    y_calib_dev = torch.tensor(y_calib_resid).to(DEVICE)

    best_calib_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for lag_b, static_b, y_b in train_loader:
            lag_b, static_b, y_b = lag_b.to(DEVICE), static_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            pred = model(lag_b, static_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            calib_pred = model(lag_calib_dev, static_calib_dev)
            calib_loss = loss_fn(calib_pred, y_calib_dev).item()
        epochs_run = epoch

        if calib_loss < best_calib_loss:
            best_calib_loss = calib_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_resid = model(lag_test.to(DEVICE), static_test.to(DEVICE)).cpu().numpy()
    pred_abs = pred_resid + test_h["pm25"].values
    y_test_actual = test_h[target_col].values

    mae = mean_absolute_error(y_test_actual, pred_abs)
    rmse = mean_squared_error(y_test_actual, pred_abs) ** 0.5
    r2 = r2_score(y_test_actual, pred_abs)

    return {
        "horizon": h, "mae": mae, "rmse": rmse, "r2": r2,
        "n_train": len(train_h), "n_calib": len(calib_h), "n_test": len(test_h),
        "epochs_run": epochs_run,
    }, model.state_dict(), scaler


def main():
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    provinces_used = sorted(df["province_en"].unique().tolist())
    print(f"จังหวัดที่ใช้เทรน ({len(provinces_used)}): {provinces_used}")

    train_df, calib_df, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  calibration={len(calib_df)}  test={len(test_df)}")

    eval_rows = []
    state_dicts = {}
    scalers = {}
    for h in range(1, 25):
        row, state_dict, scaler = train_one_horizon(h, train_df, calib_df, test_df)
        eval_rows.append(row)
        state_dicts[h] = state_dict
        scalers[h] = scaler
        print(f"t+{h:2d}  lstm: MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} R2={row['r2']:.3f} "
              f"(epochs={row['epochs_run']}, n_train={row['n_train']}, n_test={row['n_test']})")

    eval_df = pd.DataFrame(eval_rows)
    mean_mae = eval_df["mae"].mean()
    mean_rmse = eval_df["rmse"].mean()
    mean_r2 = eval_df["r2"].mean()

    print("\n" + "=" * 70)
    print(f"แถว LSTM สำหรับตารางที่ 14 (เฉลี่ยทั้ง 24 horizon, {len(provinces_used)} จังหวัดฝึกสอน):")
    print("=" * 70)
    print(f"MAE  = {mean_mae:.3f}")
    print(f"RMSE = {mean_rmse:.3f}")
    print(f"R2   = {mean_r2:.3f}")

    print("\nเทียบกับตัวเลขเดิมในตารางที่ 14:")
    print("  LightGBM        : MAE=2.780 RMSE=4.318 R2=0.939")
    print("  XGBoost         : MAE=2.778 RMSE=4.319 R2=0.938")
    print("  Persistence     : MAE=3.177 RMSE=4.910 R2=0.922")
    print("  Dynamic Routing : MAE=2.779 RMSE=4.318 R2=0.939")

    if mean_mae < MAE_SANITY_LOW or mean_mae > MAE_SANITY_HIGH:
        print(f"\n*** WARNING: MAE={mean_mae:.3f} อยู่นอกช่วงที่น่าเชื่อถือ ({MAE_SANITY_LOW}-{MAE_SANITY_HIGH}) ***")
        print("*** ตรวจสอบ scaler/split/residual sign ก่อนเอาตัวเลขนี้ไปใส่เล่ม ***")

    eval_path = os.path.join(MODEL_DIR, f"lstm_eval_by_horizon_{TIMESTAMP}.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"\nบันทึกผลดิบ (ต่อ horizon) ที่: {eval_path}")

    model_path = os.path.join(MODEL_DIR, f"lstm_pm25_multiprovince_residual_{TIMESTAMP}.pt")
    torch.save({
        "state_dicts": state_dicts, "scalers": scalers,
        "feature_cols": FEATURE_COLS, "lag_features": LAG_FEATURES, "static_features": STATIC_FEATURES,
        "trained_at": TIMESTAMP, "provinces": provinces_used, "eval_by_horizon": eval_rows,
        "architecture": "hybrid LSTM (5-step lag sequence) + dense (15 static features)",
    }, model_path)
    print(f"บันทึกโมเดลที่: {model_path}")


if __name__ == "__main__":
    main()
