import matplotlib.pyplot as plt
import numpy as np
from GMM_Gibbs import GMM_Gibbs
from sklearn.datasets import make_blobs

# 固定
# 各クラスタの共分散行列 (？)  <----- np.eye は 2 じゃなくて 3 では？
Lambda = np.array([np.eye(2) + np.eye(2) * 0.1 for _ in range(3)])
beta = np.ones(3)  # 各クラスタの分散 (？)

# データセットを生成
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# GMM_MHモデルのインスタンスを作成
model = GMM_Gibbs(K=3)

# モデルの学習
model.train(X, n_iter=2)

# 予測
pred = model.predict(X)

# 結果をプロット
plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.title("GMM via MH (Cluster Assignments with Metropolis)")
plt.show()
