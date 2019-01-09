from transformer_keras.core import PositionalEncoding
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    max_seq_len = 100
    d_model = 25

    position_encoding = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_seq_len + 1)
    ])  # [max_seq_len + 1, d_model]

    position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2])
    position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2])

    xs = np.arange(1, 100)
    plt.figure(figsize=(15, 5))
    plt.plot(xs, position_encoding[xs, 4:8])
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.savefig("../assets/pos_encoding.png")
    plt.show()


