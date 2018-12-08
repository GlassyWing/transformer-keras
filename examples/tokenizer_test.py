from transformer.tools.text_preprocess import make_dictionaries, save_dictionaries

if __name__ == '__main__':
    src_tokenizer, tgt_tokenizer = make_dictionaries("../data/en2de.s2s.all.txt",
                                                     start_token="<S>",
                                                     end_token="</S>",
                                                     oov_token="<UNK>",
                                                     filters="\t\n",
                                                     overwrite=True)

    freq_limit = 5

    src_sub = sum(map(lambda x: x[1] < freq_limit, src_tokenizer.word_counts.items()))
    tgt_sub = sum(map(lambda x: x[1] < freq_limit, tgt_tokenizer.word_counts.items()))

    src_tokenizer.num_words = len(src_tokenizer.word_index) - src_sub
    tgt_tokenizer.num_words = len(tgt_tokenizer.word_index) - tgt_sub

    print(src_tokenizer.num_words)
    print(tgt_tokenizer.num_words)

    sequnces = src_tokenizer.texts_to_sequences(
        ['a sleeping cat'])
    print(sequnces)
    texts = src_tokenizer.sequences_to_texts(sequnces)
    print(texts)

    sequnces = tgt_tokenizer.texts_to_sequences(
        ['eine schlafende Katze'])
    print(sequnces)
    texts = tgt_tokenizer.sequences_to_texts(sequnces)
    print(texts)

    save_dictionaries(src_tokenizer=src_tokenizer,
                      src_dict_path="../data/dict_en.json",
                      tgt_tokenizer=tgt_tokenizer,
                      tgt_dict_path="../data/dict_de.json", )
