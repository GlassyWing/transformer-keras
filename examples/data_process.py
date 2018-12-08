def concat(src_files, tgt_files, output_file, encoding="utf-8", sent_delimiter="\t"):

    src_sequences = []
    tgt_sequences = []

    for src_file in src_files:
        with open(src_file, encoding=encoding) as file:
            src_sequences.extend(file.readlines())

    for tgt_file in tgt_files:

        with open(tgt_file, encoding=encoding) as file:
            tgt_sequences.extend(file.readlines())

    assert len(src_sequences) == len(tgt_sequences)

    with open(output_file, mode="w+", encoding=encoding) as file:
        for src_seq, tgt_seq in zip(src_sequences, tgt_sequences):
            src_seq = src_seq.replace('\n', '')
            src_seq = src_seq.replace('-', ' @-@ ')
            tgt_seq = tgt_seq.replace('\n', '')
            tgt_seq = tgt_seq.replace('-', ' @-@ ')

            for p in '.,':
                src_seq = src_seq.replace(p, ' ' + p)
                tgt_seq = tgt_seq.replace(p, ' ' + p)



            seq = sent_delimiter.join([src_seq, tgt_seq])

            if len(seq) != 1:
                file.write(seq + "\n")


if __name__ == '__main__':
    concat(["../data/training/train.en"], ["../data/training/train.de"], "../data/en2de.s2s.txt")
    concat(["../data/validation/val.en"], ["../data/validation/val.de"], "../data/en2de.s2s.valid.txt")
    concat(["../data/validation/val.en", "../data/training/train.en"],
           ["../data/validation/val.de", "../data/training/train.de"],
           "../data/en2de.s2s.all.txt")
