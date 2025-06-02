import numpy as np
from numpy.random import dirichlet
from scipy.stats import multivariate_normal


class GMM_Gibbs:
    def __init__(self, K, sigma=1.0, mu_prior_std=5.0, alpha=1.0):
        self.K = K
        self.sigma = sigma  # 分散 (共通)
        self.mu_prior_std = mu_prior_std  # μ の事前分布の標準偏差
        self.alpha = alpha  # 事前分布 (ディリクレ分布) のパラメータ

    def initialize(self, X: np.ndarray):
        self.X = X
        self.N, self.D = X.shape
        self.mu = np.random.randn(self.K, self.D)  # K: カテゴリ数, D: 次元数
        self.z = np.random.choice(self.K, self.N)  # K: カテゴリ数, N: データ数
        self.pi = np.ones(self.K) / self.K  # 初期混合比（均等）

    def log_likelihood(self, x, mu_k):  # x がクラスタ k に属する対数尤度 p(x|mu, sigma^2)
        diff = x - mu_k
        return -0.5 * np.sum(diff**2) / self.sigma**2

    def gibbs_update_z(self):  # クラスタ割当 z_n (サイン) をギブスサンプリングで更新
        for i in range(self.N):
            log_probs = []
            for j in range(self.K):  # 各クラスタ j に対する対数尤度を計算
                log_prior = np.log(self.pi[j])
                log_likelihood = self.log_likelihood(self.X[i], self.mu[j])
                log_probs.append(log_prior + log_likelihood)

            log_probs = np.array(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)

            self.z[i] = np.random.choice(self.K, p=probs)

    def update_mu(self):  # 各クラスタの平均 μ_k を z に基づいて更新
        for i in range(self.K):
            X_k = self.X[self.z == i]
            # cov_ = np.linalg.inv(beta[k] * Lambda[k])  # <----- エラーが出たので修正
            cov_ = np.linalg.inv(
                np.eye(self.D) / (self.sigma**2) + X_k.shape[0] / self.mu_prior_std**2 * np.eye(self.D)
            )
            mean_ = X_k.mean(axis=0)
            mus_ = multivariate_normal(mean=mean_, cov=cov_).rvs(size=1)
            self.mu[i] = mus_

    def update_pi(self):  # 混合比 π をディリクレ分布からサンプリング
        counts = np.array([np.sum(self.z == i) for i in range(self.K)])
        self.pi = dirichlet(counts + self.alpha)

    # train
    def train(self, X, n_iter=100):
        self.initialize(X)
        for _ in range(n_iter):
            self.gibbs_update_z()
            self.update_mu()
            self.update_pi()

    def predict(self, X):
        assignments = []
        for x in X:
            log_probs = [np.log(self.pi[k]) + self.log_likelihood(x, self.mu[k]) for k in range(self.K)]
            log_probs = np.array(log_probs)
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            assignments.append(np.argmax(probs))
        return np.array(assignments)
