import numpy as np
from MHNG_Agent import Agent


def main():
    with open("./week3/data_a.txt", "r") as f:
        data_a = np.loadtxt(f, delimiter=" ", dtype=float)
    with open("./week3/data_b.txt", "r") as f:
        data_b = np.loadtxt(f, delimiter=" ", dtype=float)

    alpha = 0.1  # 理解していれば消してもいいです (事前分布のパラメータ)
    num_iteration = 10
    # sigma = 1.0  # <----- これ要らないのでは
    num_class = 4  # Wの種類

    agent_a = Agent(num_class, alpha, data_a)
    agent_b = Agent(num_class, alpha, data_b)
    N = len(data_a)

    init_mu_a = agent_a.mus
    init_mu_b = agent_b.mus

    # update param
    for itr in range(1, 1 + num_iteration):
        for n in range(N):
            # Agent-A send message to Agent-B
            msg_a = agent_a.sample_msg(n)
            agent_b.update_w(msg_a, n)
            agent_b.update_mu()
            # Agent-B send message to Agent-A
            msg_b = agent_b.sample_msg(n)
            agent_a.update_w(msg_b, n)
            agent_a.update_mu()

        if itr % 2 == 0:
            print("iteration:", itr)
            print("Agent-A:", agent_a.msgs)
            print("Agent-B:", agent_b.msgs)

    print("===========")
    print("mu")
    print(init_mu_a)
    print(init_mu_b)
    print("↓")
    print(agent_a.mus)
    print(agent_b.mus)


if __name__ == "__main__":
    main()
