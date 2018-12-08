from transformer.data_loader import DataLoader

if __name__ == '__main__':
    data_loader = DataLoader(src_dictionary_path="../data/dict_en.json",
                             tgt_dictionary_path="../data/dict_de.json",
                             batch_size=4)
    # X, Y = data_loader.load_data("../data/en2de.s2s.txt")
    # print(X.shape)
    # print(Y.shape)
    generator = data_loader.generator("../data/en2de.s2s.txt")

    for _ in range(1):
        x, _ = next(generator)
        print(data_loader.src_tokenizer.sequences_to_texts(x[0]))
        print(data_loader.tgt_tokenizer.sequences_to_texts(x[1]))

    data_loader.dump_to_h5("../data/en2de.s2s.txt", "../data/en2de.s2s.h5")
    data_loader.dump_to_h5("../data/en2de.s2s.valid.txt", "../data/en2de.s2s.valid.h5")
