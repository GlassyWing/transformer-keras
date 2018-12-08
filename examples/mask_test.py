import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from transformer.core import sequence_mask, padding_mask
from transformer.tools.text_preprocess import load_dictionary

if __name__ == '__main__':
    tokenizer = load_dictionary("../data/dict_en.json")
    sequence = tokenizer.texts_to_sequences(['Two young , White males are outside near many bushes .'])

    sequence = pad_sequences(sequence, maxlen=15, padding="post")

    print(sequence)
    pad_mask = K.get_value(padding_mask(sequence, sequence))
    seq_mask = K.get_value(sequence_mask(sequence))

    # 1 indicates that retention is required
    print(pad_mask[0])
    print(seq_mask[0])

    mask = np.minimum(pad_mask, seq_mask)

    plt.figure(figsize=(5, 5))
    plt.imshow(mask[0])
    plt.savefig('../assets/mask.png')
    plt.show()

