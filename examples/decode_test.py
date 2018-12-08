import time

from transformer import get_or_create
from transformer.tools.text_preprocess import load_dictionaries


def print_result(result):
    if type(result) == list:
        for seq, score in result:
            print(seq, score)
    else:
        print(result)


if __name__ == '__main__':
    en2de_translator = get_or_create("../data/default-config.json",
                                     weights_path="../models/weights.32-1.38.h5")

    src_tokenizer, tgt_tokenizer = load_dictionaries("../data/dict_en.json",
                                                     "../data/dict_de.json")

    sentences = [
        'A brunette woman is standing on the sidewalk looking down the road .',
        'A group of three friends are conversing inside of a home .',
        'Two chinese people are standing by a chalkboard .',
        'A person wearing blue jeans and a red sweater us turning the corner of a brick wall .',
        'Farmers are performing their agriculture during the day .',
        'At some sort of carnival, a man is making cotton candy .',
        'A bunch of police officers are standing outside a bus .',
        'A elderly white @-@ haired woman is looking inside her register and looking through her glasses .',
        'Two men are standing at telephone booths outside .',
        'Two women wearing red and a man coming out of a port @-@ a @-@ potty .',
    ]

    # Preheating model
    en2de_translator.beam_search_text_decode(
        ['Two women wearing red and a man coming out of a port @-@ a @-@ potty .'],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        delimiter=' ',
    )

    total_time = 0.0
    for sent in sentences:
        start_time = time.time()
        result = en2de_translator.beam_search_text_decode(
            [sent],
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            delimiter=' ',
        )
        t = (time.time() - start_time) * 1000
        total_time += t

    print_result(result)

    print(f"mean cost {(total_time / len(sentences))} ms")
