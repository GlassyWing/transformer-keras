import json

import h5py
from keras.preprocessing.sequence import pad_sequences

from transformer.tools.text_preprocess import tokenizer_from_json


class DataLoader:

    def __init__(self, src_dictionary_path,
                 tgt_dictionary_path,
                 word_delimiter=' ',
                 sent_delimiter='\t',
                 max_len=999,
                 batch_size=64):

        with open(src_dictionary_path, encoding="utf-8") as f:
            self.src_tokenizer = tokenizer_from_json(json.load(f))

        with open(tgt_dictionary_path, encoding="utf-8") as f:
            self.tgt_tokenizer = tokenizer_from_json(json.load(f))

        self.word_delimiter = word_delimiter
        self.sent_delimiter = sent_delimiter
        self.max_len = max_len
        self.batch_size = batch_size
        self.src_vocab_size = min(self.src_tokenizer.num_words, len(self.src_tokenizer.index_word))
        self.tgt_vocab_size = min(self.tgt_tokenizer.num_words, len(self.tgt_tokenizer.index_word))

    def __pad_seq(self, src_sents, tgt_sents):
        src_sents = self.src_tokenizer.texts_to_sequences(src_sents)
        tgt_sents = self.tgt_tokenizer.texts_to_sequences(tgt_sents)

        src_max_len = min(len(max(src_sents, key=len)), self.max_len)
        tgt_max_len = min(len(max(tgt_sents, key=len)), self.max_len)

        src_sents = pad_sequences(src_sents, maxlen=src_max_len, padding='post')
        tgt_sents = pad_sequences(tgt_sents, maxlen=tgt_max_len, padding='post')
        return src_sents, tgt_sents

    def load_txt_data(self, txt_file):
        sents = [[], []]
        for sent in self.load_sents(txt_file):
            for seq, xs in zip(sent, sents):
                xs.append(seq.split(self.word_delimiter))

        src_sents, tgt_sents = sents[0], sents[1]
        src_sents, tgt_sents = self.__pad_seq(src_sents, tgt_sents)
        return src_sents, tgt_sents

    def load_h5_data(self, h5_file):
        with h5py.File(h5_file) as file:
            X, Y = file['X'][:], file['Y'][:]
        return X, Y

    def dump_to_h5(self, txt_file, h5_path):
        src_sents, tgt_sents = self.load_txt_data(txt_file)
        with h5py.File(h5_path, 'w') as file:
            file.create_dataset('X', data=src_sents)
            file.create_dataset('Y', data=tgt_sents)

    def generator(self, txt_file):
        while True:
            sents = [[], []]
            for sent in self.load_sents(txt_file):
                for seq, xs in zip(sent, sents):
                    xs.append(seq.split(self.word_delimiter))
                if len(sents[0]) >= self.batch_size:
                    src_sents, tgt_sents = sents[0], sents[1]
                    src_sents, tgt_sents = self.__pad_seq(src_sents, tgt_sents)
                    yield [src_sents, tgt_sents], None
                    sents = [[], []]

    def load_sents(self, txt_file):
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                yield line.rstrip('\r\n').split(self.sent_delimiter)
