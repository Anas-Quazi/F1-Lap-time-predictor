import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.split_data import make_grouped_splits


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/features/telemetry_features.csv")

# avoid bad rows
df = df.dropna(subset=["laptime"]).copy()

# split
train_df, val_df, test_df = make_grouped_splits(df, target_col="laptime")

# target and features
target_col = "laptime"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].astype(np.float32)

X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col].astype(np.float32)

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].astype(np.float32)

# ----------------------------
# Select columns
# ----------------------------
# We use only numeric features for now
# You can later add embedding features for driver/team/compound if needed
cat_cols = []
num_cols = [c for c in X_train.columns if c not in ["race", "driver", "team", "raceid", "year"] and pd.api.types.is_numeric_dtype(X_train[c])]
X_train_num = X_train[num_cols].copy()
X_val_num = X_val[num_cols].copy()
X_test_num = X_test[num_cols].copy()

# fill NaN
for col in num_cols:
    X_train_num[col] = X_train_num[col].fillna(0)
    X_val_num[col] = X_val_num[col].fillna(0)
    X_test_num[col] = X_test_num[col].fillna(0)

# scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)
X_test_num = scaler.transform(X_test_num)

X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
X_val_num = torch.tensor(X_val_num, dtype=torch.float32)
X_test_num = torch.tensor(X_test_num, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# ----------------------------
# ANN model
# ----------------------------
class F1ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = F1ANN(X_train_num.shape[1])

train_loader = DataLoader(
    TensorDataset(X_train_num, y_train),
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_num, y_val),
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test_num, y_test),
    batch_size=128,
    shuffle=False
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
best_state = None

# ----------------------------
# training loop
# ----------------------------
for epoch in range(80):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * len(xb)

    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

model.load_state_dict(best_state)

# ----------------------------
# evaluation
# ----------------------------
def evaluate(loader):
    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            preds.extend(pred.cpu().numpy())
            true.extend(yb.cpu().numpy())

    preds = np.array(preds)
    true = np.array(true)

    mae = mean_absolute_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))
    r2 = r2_score(true, preds)

    return mae, rmse, r2

mae, rmse, r2 = evaluate(test_loader)

print("Test MAE:", round(mae, 4))
print("Test RMSE:", round(rmse, 4))
print("Test R2:", round(r2, 4))