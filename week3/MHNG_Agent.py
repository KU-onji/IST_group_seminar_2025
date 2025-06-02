import numpy as np
from numpy.random import dirichlet


class Agent:
    def __init__(self, msg_D, alpha, data):
        self.K = msg_D
        self.alpha = alpha
        self.data = data
        self.msgs = np.random.randint(0, self.K, len(data))  # W

        # init update params
        counts = np.array([np.sum(self.msgs == i) for i in range(self.K)])
        self.prior2 = dirichlet(counts + self.alpha)

        self.update_mu()
        print(self.msgs)

    def calc_lik(self, x, mu):
        D = len(x)
        diff = x - mu
        gaussian = np.exp(-0.5 * np.sum(diff**2) / D) / np.sqrt((2 * np.pi) ** D)  # 正規分布
        return gaussian

    def update_w(self, w_new, n):
        w_current = self.msgs[n]
        probs = [self.calc_lik(self.data[n], self.mus[i]) * self.prior2[i] for i in range(self.K)]
        r = min(1.0, probs[w_new] / probs[w_current])
        u = np.random.rand()
        if u < r:
            self.msgs[n] = w_new
            counts = np.array([np.sum(self.msgs == i) for i in range(self.K)])
            self.prior2 = dirichlet(counts + self.alpha)

    def update_mu(self):
        self.mus = []
        for k in range(self.K):
            X_k = self.data[self.msgs == k]  # クラスタ k に属するデータ
            if len(X_k) > 0:
                self.mus.append(X_k.mean(axis=0))  # クラスタ k の平均
            else:
                # クラスタ k にデータがない場合はランダムに初期化
                self.mus.append(self.data[np.random.choice(len(self.data))])
        self.mus = np.array(self.mus)

    def sample_msg(self, n):
        # p(w|x)
        probs = [self.calc_lik(self.data[n], self.mus[i]) * self.prior2[i] for i in range(self.K)]
        # p(w|x) から サンプルの生成
        new_msg = np.random.choice(self.K, p=probs / np.sum(probs))
        return new_msg
