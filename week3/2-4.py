import matplotlib.pyplot as plt
import numpy as np


def target_distribution(theta):
    return np.exp(-(theta**2) / 2) / np.sqrt(2 * np.pi)


def metropolis_hastings_uniform(num_samples, proposal_width=1.0, burn_in=100):
    samples = []
    theta = np.random.randn()  # 初期値
    for _ in range(num_samples + burn_in):
        # 提案： 一様分布 Q(θ'|θ) ~ U[θ - w, θ + w]
        theta_new = theta + np.random.uniform(-proposal_width, proposal_width)

        p_current = target_distribution(theta)
        p_new = target_distribution(theta_new)

        alpha = min(1, p_new / p_current)
        u = np.random.rand()

        if u < alpha:
            theta = theta_new
        samples.append(theta)
    return np.array(samples[burn_in:])  # burn-in分を除外


# 実行
samples = metropolis_hastings_uniform(num_samples=10000, proposal_width=1.0, burn_in=500)

# 可視化
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label="MH samples (Uniform Proposal)")
plt.title("Metropolis-Hastings with Uniform Proposal")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.show()
