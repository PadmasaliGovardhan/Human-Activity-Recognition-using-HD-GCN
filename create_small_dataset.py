import numpy as np
import os

# -------------------------
# Small NTU-like parameters
# -------------------------
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 20
C, T, V, M = 3, 30, 25, 1   # coords, frames, joints, persons

TOTAL = NUM_CLASSES * SAMPLES_PER_CLASS
D = V * C * M  # flattened joint dimension (75)

# -------------------------
# Generate synthetic skeletons
# Model-native shape: (N, C, T, V, M)
# -------------------------
x = np.random.randn(TOTAL, C, T, V, M).astype(np.float32)

# -------------------------
# Convert to NTU feeder format
# Required stored shape: (N, T, D)
# -------------------------
x = x.transpose(0, 2, 3, 1, 4)   # N, T, V, C, M
x = x.reshape(TOTAL, T, D)      # N, T, D

# -------------------------
# One-hot labels (NTU-style)
# Shape: (N, num_class)
# -------------------------
y = np.zeros((TOTAL, NUM_CLASSES), dtype=np.int64)
idx = 0
for cls in range(NUM_CLASSES):
    for _ in range(SAMPLES_PER_CLASS):
        y[idx, cls] = 1
        idx += 1

# -------------------------
# Train / Test split (single file)
# -------------------------
split = int(0.8 * TOTAL)
x_train, y_train = x[:split], y[:split]
x_test,  y_test  = x[split:], y[split:]

# -------------------------
# Save ONE NTU-style NPZ
# -------------------------
os.makedirs("data/ntu_small", exist_ok=True)
np.savez(
    "data/ntu_small/NTU_SMALL.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print("✅ NTU-faithful small dataset created")
print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape,  y_test.shape)
