import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# 1. นำภาพใบหน้าจาก Folder data มาแปลงเป็น Vector
data_dir = 'data'
files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heif'))]
X, filenames = [], []

for f in files:
    try:
        img = Image.open(os.path.join(data_dir, f)).convert('L').resize((64, 64))
        X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        filenames.append(f)
    except Exception:
        pass

X = np.array(X)

# ลดมิติข้อมูลเป็น 2D ด้วย SVD (PCA) เพื่อให้ฟังก์ชัน plot() ใช้งานได้
X_mean = np.mean(X, axis=0)
_, _, Vt = np.linalg.svd(X - X_mean, full_matrices=False)
X_2d = np.dot(X - X_mean, Vt[:2].T)

def plot(C, idx=None):
    plt.clf()
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    if idx is not None:
        for k in range(K):
            plt.plot(X_2d[idx == k, 0], X_2d[idx == k, 1], '.' + colors[k % len(colors)], label=f'Cluster {k+1}')
        plt.legend()
    else:
        plt.plot(X_2d[:, 0], X_2d[:, 1], '.k')
    C_2d = np.dot(C - X_mean, Vt[:2].T)
    plt.plot(C_2d[:, 0], C_2d[:, 1], 'sk', markersize=8)
    plt.title('Face Similarity Detection (Fuzzy C-Means Clustering)')
    plt.pause(0.5)

K = 3
m = 2
C = np.zeros((K, X.shape[1]))

mu = np.random.rand(len(X), K)
while True:
    mu /= np.sum(mu, axis=1)[:, None]
    Cold = C.copy()
    for k in range(K):
        C[k] = np.dot(mu[:, k] ** m, X) / np.sum(mu[:, k] ** m)
        D = np.sqrt(np.sum((X - C[k])**2, axis=1))   # Euclidean distance
        D = np.fmax(D, 1e-8) ** (-2/(m-1))
        mu[:, k] = D
    idx = np.argmax(mu, axis=1)
    plot(C, idx)
    if np.mean(np.abs(Cold-C)) < 1e-4:
        print('finished')
        break

# แสดงผลการจัดกลุ่มใบหน้า
idx = np.argmax(mu, axis=1)
for k in range(K):
    members = [filenames[i] for i in range(len(filenames)) if idx[i] == k]
    print(f"\nCluster {k+1} ({len(members)} images):")
    for m in members:
        print(f"  - {m}")
plt.show()