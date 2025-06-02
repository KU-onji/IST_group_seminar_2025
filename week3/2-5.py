import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def target_distribution(theta, phi):
    return norm.pdf(theta) * norm.pdf(phi)  # 二次元正規分布の確率密度関数


def metropolis_hastings_gaussian(num_samples, proposal_width=None, burn_in=100):
    samples = []
    theta, phi = np.random.randn(2)  # 初期値

    for _ in range(num_samples):
        if proposal_width is None:
            # 提案： 正規分布 Q(θ'|θ) ~ N(θ, φ^2)
            theta_new = theta + np.random.randn()
            phi_new = phi + np.random.randn()
        else:
            # 提案： 一様分布 Q(θ'|θ) ~ U[θ - w, θ + w]
            theta_new = theta + np.random.uniform(-proposal_width, proposal_width)
            phi_new = phi + np.random.uniform(-proposal_width, proposal_width)

        p_current = target_distribution(theta, phi)
        p_new = target_distribution(theta_new, phi_new)

        alpha = min(1, p_new / p_current)
        u = np.random.rand()

        if u < alpha:
            theta = theta_new
            phi = phi_new
        samples.append((theta, phi))
    return np.array(samples[burn_in:])  # burn-in分を除外


# 実行 (提案幅なしで正規分布を使用)
samples = metropolis_hastings_gaussian(num_samples=10000, proposal_width=None, burn_in=500)

# 実行 (提案幅ありで一様分布を使用)
samples_uniform = metropolis_hastings_gaussian(num_samples=10000, proposal_width=1.0, burn_in=500)

# プロット
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.title("Samples from Metropolis-Hastings")
plt.xlabel("$\\theta$")
plt.ylabel("$\\phi$")
plt.axis("equal")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.hist2d(samples[:, 0], samples[:, 1], bins=50, density=True, cmap="viridis")
plt.title("2D Histogram")
plt.xlabel("$\\theta$")
plt.ylabel("$\\phi$")
plt.colorbar()
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.scatter(samples_uniform[:, 0], samples_uniform[:, 1], s=5, alpha=0.5)
plt.title("Samples from Metropolis-Hastings (Uniform Proposal)")
plt.xlabel("$\\theta$")
plt.ylabel("$\\phi$")
plt.axis("equal")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist2d(samples_uniform[:, 0], samples_uniform[:, 1], bins=50, density=True, cmap="viridis")
plt.title("2D Histogram (Uniform Proposal)")
plt.xlabel("$\\theta$")
plt.ylabel("$\\phi$")
plt.colorbar()
plt.tight_layout()
plt.show()
