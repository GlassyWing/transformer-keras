import json
import os
from keras_preprocessing.text import Tokenizer, text_to_word_sequence


class CustomTokenizer(Tokenizer):

    def __init__(self,
                 start_token=None,
                 end_token=None,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 document_count=0,
                 **kwargs):

        self.start_token = start_token
        self.end_token = end_token

        super().__init__(num_words,
                         filters,
                         lower,
                         split,
                         char_level,
                         oov_token,
                         document_count,
                         **kwargs)

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.

                In the case where texts contains lists,
                we assume each entry of the lists to be a token.

                Required before using `texts_to_sequences` or `texts_to_matrix`.

                # Arguments
                    texts: can be a list of strings,
                        a generator of strings (for memory-efficiency),
                        or a list of list of strings.
                """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        sorted_voc = []

        if self.start_token is not None:
            sorted_voc.append(self.start_token)

        if self.end_token is not None:
            sorted_voc.append(self.end_token)

        if self.oov_token is not None:
            sorted_voc.append(self.oov_token)

        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        end_token_index = self.word_index.get(self.end_token)
        start_token_index = self.word_index.get(self.start_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            vect = []
            if self.start_token is not None:
                vect.append(start_token_index)

            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            if self.end_token is not None:
                vect.append(end_token_index)
            yield vect

    def get_config(self):
        config = super().get_config()
        config['start_token'] = self.start_token
        config['end_token'] = self.end_token
        return config


def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.

    # Arguments
        json_string: JSON string encoding a tokenizer configuration.

    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = CustomTokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer


def load_dictionary(dict_path, encoding="utf-8"):
    with open(dict_path, encoding=encoding, mode="r") as file:
        return tokenizer_from_json(json.load(file))


def save_dictionary(tokenizer, dict_path, encoding="utf-8"):
    with open(dict_path, mode="w+", encoding=encoding) as file:
        json.dump(tokenizer.to_json(), file)


def save_dictionaries(src_tokenizer, src_dict_path,
                      tgt_tokenizer, tgt_dict_path,
                      encoding="utf-8"):
    save_dictionary(src_tokenizer, src_dict_path, encoding)
    save_dictionary(tgt_tokenizer, tgt_dict_path, encoding)


def load_dictionaries(src_dict_path,
                      tgt_dict_path,
                      encoding="utf-8"):
    src_tokenizer = load_dictionary(src_dict_path, encoding)
    tgt_tokenizer = load_dictionary(tgt_dict_path, encoding)
    return src_tokenizer, tgt_tokenizer


def make_dictionaries(file_path,
                      src_dict_path=None,
                      tgt_dict_path=None,
                      sent_delimiter='\t',
                      start_token='<S>',
                      end_token='</S>',
                      oov_token='<UNK>',
                      filters="\t\n",
                      overwrite=False,
                      encoding="utf-8",
                      **kwargs):
    """

    :param end_token: end token
    :param encoding: file encoding
    :param overwrite: overwrite the existing file
    :param file_path: corpus file path
    :param src_dict_path:   the path to save source dictionary
    :param tgt_dict_path:   the path to save target dictionary
    :param sent_delimiter:  delimiter between source and target sentences.
    :param start_token:     start token
    :param oov_token:       unkown token
    :param filters:         character which should be ignore.
    :return:
    """
    with open(file_path, encoding=encoding) as f:
        src_sents = []
        tgt_sents = []
        for line in f.readlines():
            sent = line.split(sent_delimiter)
            src_sents.append(sent[0])
            tgt_sents.append(sent[1])

    if src_dict_path is not None and os.path.exists(src_dict_path) and not overwrite:
        with open(src_dict_path, encoding=encoding, mode="r") as file:
            src_tokenizer = tokenizer_from_json(json.load(file))
    else:
        src_tokenizer = CustomTokenizer(start_token=start_token,
                                        end_token=end_token,
                                        oov_token=oov_token,
                                        filters=filters, **kwargs)
        src_tokenizer.fit_on_texts(src_sents)
        if src_dict_path is not None:
            with open(src_dict_path, encoding=encoding, mode="w+") as f:
                json.dump(src_tokenizer.to_json(), f)

    if tgt_dict_path is not None and os.path.exists(tgt_dict_path) and not overwrite:
        with open(tgt_dict_path, encoding=encoding, mode="r") as file:
            tgt_tokenizer = tokenizer_from_json(json.load(file))
    else:
        tgt_tokenizer = CustomTokenizer(start_token=start_token,
                                        end_token=end_token,
                                        oov_token=oov_token,
                                        filters=filters, **kwargs)
        tgt_tokenizer.fit_on_texts(tgt_sents)
        if tgt_dict_path is not None:
            with open(tgt_dict_path, encoding=encoding, mode="w+") as f:
                json.dump(tgt_tokenizer.to_json(), f)

    return src_tokenizer, tgt_tokenizer
