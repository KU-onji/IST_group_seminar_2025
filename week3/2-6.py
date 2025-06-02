import matplotlib.pyplot as plt
import numpy as np


def gibbs_sampler(num_samples):
    x_samples, y_samples = np.zeros(num_samples), np.zeros(num_samples)
    x, y = 0, 0  # 初期値

    for i in range(num_samples):
        # 条件付き分布からサンプリング
        x = np.random.normal(-y / 2, 1)  # P(X|Y)
        y = np.random.normal(-x / 2, 1)  # P(Y|X)

        x_samples[i] = x
        y_samples[i] = y

    return x_samples, y_samples


# 実行
x_samples, y_samples = gibbs_sampler(num_samples=10000)

# 可視化
samples = np.array([x_samples, y_samples])
cov = np.cov(samples)
corr = np.corrcoef(samples)
plt.scatter(x_samples, y_samples, alpha=0.5)
plt.title("Gibbs Sampling Results")
plt.xlabel("X samples")
plt.ylabel("Y samples")
plt.text(-1.5, -0.5, f"Covariance:\n{cov}\nCorrelation:\n{corr}", fontsize=7, bbox=dict(facecolor="white", alpha=0.5))
plt.show()
