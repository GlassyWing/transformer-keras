import json

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.engine import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Dropout, Lambda, Softmax, Dense, Add, Embedding, Conv1D
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from transformer_keras.tools.text_preprocess import CustomTokenizer, load_dictionary


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention:

    def __init__(self, attention_dropout=0.0):
        self.attention_dropout = attention_dropout

    def __call__(self, q, k, v, attn_mask=None, scale=1.0):
        """

        :param q: Queries 张量，形状为[N, T_q, D_q]
        :param k: Keys 张量，形状为[N, T_k, D_k]
        :param v: Values 张量，形状为[N, T_v, D_v]
        :param attn_mask: 注意力掩码，形状为[N, T_q, T_k]
        :param scale: 缩放因子，浮点标量
        :return: 上下文张量和注意力张量
        """

        attention = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2, 2)) * scale)([q, k])  # [N, T_q, T_k]
        if attn_mask is not None:
            # 为需要掩码的地方设置一个负无穷，softmax之后就会趋近于0
            attention = Lambda(lambda x: (-1e+10) * (1 - x[0]) + x[1])([attn_mask, attention])
        attention = Softmax(axis=-1)(attention)
        attention = Dropout(self.attention_dropout)(attention)  # [N, T_q, T_k]
        context = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2, 1)))([attention, v])  # [N, T_q, D_q]
        return context, attention


def _split(x, dim_per_head, num_heads):
    shape = K.shape(x)  # [N, T_q, dim_per_head * num_heads]
    x = K.reshape(x, (shape[0], shape[1], num_heads, dim_per_head,))
    x = K.permute_dimensions(x, pattern=(0, 2, 1, 3))  # [N, num_heads, T_q, dim_per_head]
    x = K.reshape(x, (-1, shape[1], dim_per_head))  # [N * num_heads, T_q, dim_per_head]
    return x


def _concat(x, dim_per_head, num_heads):
    shape = K.shape(x)  # [N * num_heads, T_q, dim_per_head]
    x = K.reshape(x, (-1, num_heads, shape[1], dim_per_head))  # [N, num_heads, T_q, dim_per_head]
    x = K.permute_dimensions(x, [0, 2, 1, 3])  # [N, T_q, num_heads, dim_per_head]
    x = K.reshape(x, (-1, shape[1], num_heads * dim_per_head))  # [N, T_q, num_heads * dim_per_head]
    return x


class MultiHeadAttention:

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_v = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_q = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_final = Dense(model_dim, use_bias=False)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
        self.split = Lambda(lambda x: _split(x, self.dim_per_head, self.num_heads),
                            output_shape=(None, self.dim_per_head))
        self.concat = Lambda(lambda x: _concat(x, self.dim_per_head, self.num_heads),
                             output_shape=(None, self.num_heads * self.dim_per_head))

    def __call__(self, query, key, value, attn_mask=None):
        """

        :param query: shape of [N, T_q, D_q]
        :param key: shape of [N, T_k, D_k]
        :param value: shape of [N, T_v, D_v]
        :param attn_mask: shape of [N, T_q, T_k]
        :return:
        """

        residual = query

        # In order to reduce information loss, linear projection is used
        query = self.linear_q(query)  # [N, T_q, dim_per_head * num_heads]
        key = self.linear_k(key)
        value = self.linear_v(value)

        # It is divided into num_heads parts
        query = self.split(query)  # [N * num_heads, T_q, dim_per_head]
        key = self.split(key)
        value = self.split(value)

        if attn_mask is not None:
            attn_mask = Lambda(lambda x: K.repeat_elements(x, self.num_heads, axis=0))(attn_mask)
        scale = self.dim_per_head ** -0.5
        context, attention = self.dot_product_attention(query, key, value, attn_mask, scale)
        context = self.concat(context)
        context = self.linear_final(context)

        # dropout
        output = self.dropout(context)

        # add residual
        output = Add()([residual, output])

        # apply layer normalize
        output = self.layer_norm(output)

        return output, attention


def padding_mask(seq_q, seq_k):
    """
    A sentence is filled with 0, which is not what we need to pay attention to
    :param seq_k: shape of [N, T_k], T_k is length of sequence
    :param seq_q: shape of [N, T_q]
    :return: a tensor with shape of [N, T_q, T_k]
    """

    q = K.expand_dims(K.ones_like(seq_q, dtype="float32"), axis=-1)  # [N, T_q, 1]
    k = K.cast(K.expand_dims(K.not_equal(seq_k, 0), axis=1), dtype='float32')  # [N, 1, T_k]
    return K.batch_dot(q, k, axes=[2, 1])


def sequence_mask(seq):
    """

    :param seq: shape of [N, T_q]
    :return:
    """
    seq_len = K.shape(seq)[1]
    batch_size = K.shape(seq)[:1]
    return K.cast(K.cumsum(tf.eye(seq_len, batch_shape=batch_size), axis=1), dtype='float32')


class PositionalEncoding:

    def __init__(self, max_seq_len, d_model=512):
        """

        :param d_model:
        :param max_seq_len:
        """

        position_encoding = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_seq_len + 1)
        ])  # [max_seq_len + 1, d_model]

        position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2])
        position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2])

        self.position_encoding = Embedding(max_seq_len + 1, d_model,
                                           weights=[position_encoding], trainable=False)

    def __call__(self, x):
        """

        :param x: a tensor with shape of [N, max_seq_len]
        :return: position encoding
        """
        pos_seq = Lambda(self.get_pos_seq)(x)
        return self.position_encoding(pos_seq)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), dtype="int32")
        pos = K.cumsum(K.ones_like(x, dtype='int32'), axis=1)
        return mask * pos


class PositionalWiseFeedForward:

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        self.w1 = Conv1D(ffn_dim, kernel_size=1, activation="relu")
        self.w2 = Conv1D(model_dim, kernel_size=1)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def __call__(self, x):
        """

        :param x: a tensor with shape of [N, T_q, D_q]
        :return:
        """

        output = self.w2(self.w1(x))
        output = self.dropout(output)
        output = self.layer_norm(Add()([output, x]))
        return output


class EncoderLayer:

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def __call__(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder:

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        self.encoder_layers = [EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
                               for _ in range(num_layers)]
        self.seq_embedding = Embedding(vocab_size + 1, model_dim)
        self.pos_embedding = PositionalEncoding(max_seq_len, model_dim)

    def __call__(self, inputs):
        seq_emb = self.seq_embedding(inputs)
        pos_emb = self.pos_embedding(inputs)

        output = Add()([seq_emb, pos_emb])

        self_attention_mask = Lambda(lambda x: padding_mask(x, x))(inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class DecoderLayer:

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def __call__(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        """

        :param dec_inputs: [N, T_t, dim_model]
        :param enc_outputs: [N, T_s, dim_model]
        :param self_attn_mask: [N, T_t, T_t]
        :param context_attn_mask: [N, T_t, T_s]
        :return:
        """

        # self attention, all inputs are decoder inputs , [N, T_t, dim_model]
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's inputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(dec_output, enc_outputs, enc_outputs, context_attn_mask)

        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder:

    def __init__(self, vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        self.num_layers = num_layers
        self.decoder_layers = [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        self.seq_embedding = Embedding(vocab_size + 1, model_dim)
        self.pos_embedding = PositionalEncoding(max_seq_len, model_dim)

    def __call__(self, inputs, enc_output, context_attn_mask=None):
        """

        :param inputs: [N, T_t]
        :param enc_output: [N, T_s, dim_model]
        :param context_attn_mask: [N, T_t, T_s]
        :return:
        """

        seq_emb = self.seq_embedding(inputs)  # [N, T_t, dim_model]
        pos_emb = self.pos_embedding(inputs)

        output = Add()([seq_emb, pos_emb])  # [N, T_t, dim_model]

        self_attention_padding_mask = Lambda(lambda x: padding_mask(x, x),
                                             name="self_attention_padding_mask")(inputs)  # [N, T_t, T_t]
        seq_mask = Lambda(lambda x: sequence_mask(x),
                          name="sequence_mask")(inputs)
        # self_attn_mask = Add(name="self_attn_mask")([self_attention_padding_mask, seq_mask])  # [N, T_t, T_t]
        self_attn_mask = Lambda(lambda x: K.minimum(x[0], x[1]))(
            [self_attention_padding_mask, seq_mask])  # [N, T_t, T_t]

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


def _get_loss(args):
    y_pred, y_true = args
    y_true = tf.cast(y_true, 'int32')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    loss = K.mean(loss)
    return loss


def _get_accuracy(args):
    y_pred, y_true = args
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
    corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
    return K.mean(corr)


class Transformer:
    __singleton = None

    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 optimizer=Adam(lr=1e-3),
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2,
                 src_tokenizer=None,
                 tgt_tokenizer=None,
                 weights_path=None):

        self.optimizer = optimizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.decode_model = None  # used in beam_search
        self.encode_model = None  # used in beam_search
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.linear = Dense(tgt_vocab_size + 1, use_bias=False)
        self.softmax = Softmax(axis=2)

        self.pred_model, self.model = self.__build_model()
        if weights_path is not None:
            self.model.load_weights(weights_path)

    def get_config(self):
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "model_dim": self.model_dim,
            "src_max_len": self.src_max_len,
            "tgt_max_len": self.tgt_max_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout
        }

    def __build_model(self):
        src_seq_input = Input(shape=(None,), name="src_seq_input", dtype='int32')
        tgt_seq_input = Input(shape=(None,), name="tgt_seq_input", dtype='int32')

        src_seq = src_seq_input
        tgt_seq = Lambda(lambda x: x[:, :-1], name="tgt_seq")(tgt_seq_input)
        tgt_true = Lambda(lambda x: x[:, 1:], name="tgt_true")(tgt_seq_input)

        context_attn_mask = Lambda(lambda x: padding_mask(x[0], x[1]),
                                   name="context_attn_mask")([tgt_seq, src_seq])  # (N, T_t, T_s)
        enc_output, enc_self_attn = self.encoder(src_seq)  # (N, T_s, dim_model)

        output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, enc_output, context_attn_mask)
        final_output = self.linear(output)
        y_pred = self.softmax(final_output)

        loss = Lambda(_get_loss, name="loss")([final_output, tgt_true])
        ppl = Lambda(K.exp)(loss)
        accuracy = Lambda(_get_accuracy, name="accuracy")([final_output, tgt_true])

        pred_model = Model([src_seq_input, tgt_seq_input], y_pred)

        train_model = Model([src_seq_input, tgt_seq_input], loss)
        train_model.add_loss([loss])

        train_model.compile(self.optimizer, None)
        train_model.metrics_names.append('ppl')
        train_model.metrics_tensors.append(ppl)
        train_model.metrics_names.append('accuracy')
        train_model.metrics_tensors.append(accuracy)

        return pred_model, train_model

    def decode_sequence(self, input_seq, tgt_tokenizer: CustomTokenizer,
                        delimiter=' '):
        assert self.tgt_tokenizer is not None

        src_seq = self.seq_to_matrix(input_seq)
        target_seq = np.zeros((1, self.tgt_max_len))

        end_token_id = tgt_tokenizer.word_index[tgt_tokenizer.end_token]
        decoded_tokens = []

        # Use start token as activate value
        target_seq[0, 0] = tgt_tokenizer.word_index[tgt_tokenizer.start_token]

        for i in range(self.tgt_max_len - 1):
            output = self.pred_model.predict_on_batch([src_seq, target_seq])
            cur_idx = np.argmax(output[0, i, :])
            cur_token = tgt_tokenizer.index_word[cur_idx]
            decoded_tokens.append(cur_token)
            if cur_idx == end_token_id: break
            target_seq[0, i + 1] = cur_idx
        return delimiter.join(decoded_tokens[: -1])

    def decode_text(self, texts, delimiter=' '):
        assert self.src_tokenizer is not None
        sequences = self.src_tokenizer.texts_to_sequences(texts)
        return self.decode_sequence(sequences, self.tgt_tokenizer, delimiter)

    def seq_to_matrix(self, input_seq):
        max_len = min(len(max(input_seq, key=len)), self.src_max_len)
        return pad_sequences(input_seq, maxlen=max_len, padding='post')

    def make_fast_decode_model(self):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')
        src_seq = src_seq_input
        tgt_seq = tgt_seq_input

        enc_output, _ = self.encoder(src_seq)
        self.encode_model = Model(src_seq_input, enc_output)

        context_attn_mask = Lambda(lambda x: padding_mask(x[0], x[1]),
                                   name="context_attn_mask")([tgt_seq, src_seq])  # (N, T_t, T_s)

        enc_ret_input = Input(shape=(None, self.model_dim))
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, context_attn_mask)
        final_output = self.linear(dec_output)
        final_output = self.softmax(final_output)
        self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)

        self.encode_model.compile('adam', 'mse')
        self.decode_model.compile('adam', 'mse')

    def decode_sequence_fast(self, input_seq, delimiter=' '):
        assert self.tgt_tokenizer is not None

        if self.decode_model is None: self.make_fast_decode_model()
        src_seq = self.seq_to_matrix(input_seq)
        enc_output = self.encode_model.predict_on_batch(src_seq)

        tgt_tokenizer = self.tgt_tokenizer

        start_token_id = tgt_tokenizer.word_index[tgt_tokenizer.start_token]
        end_token_id = tgt_tokenizer.word_index[tgt_tokenizer.end_token]

        target_seq = np.zeros((1, self.tgt_max_len))
        target_seq[0, 0] = start_token_id

        decoded_tokens = []

        for i in range(self.tgt_max_len - 1):
            dec_output = self.decode_model.predict_on_batch([src_seq, enc_output, target_seq])
            cur_index = np.argmax(dec_output[0, i, :])
            cur_token = tgt_tokenizer.index_word[cur_index]
            if cur_index == end_token_id: break
            decoded_tokens.append(cur_token)
            target_seq[0, i + 1] = cur_index
        return delimiter.join(decoded_tokens)

    def decode_text_fast(self, texts, delimiter=' '):
        assert self.src_tokenizer is not None

        sequences = self.src_tokenizer.texts_to_sequences(texts)
        return self.decode_sequence_fast(sequences, delimiter)

    def beam_search_sequence_decode(self, input_seq,
                                    topk=5, delimiter=' '):
        assert len(input_seq) == 1  # Only one sequence is currently supported
        assert self.tgt_tokenizer is not None

        if self.decode_model is None: self.make_fast_decode_model()
        src_seq = self.seq_to_matrix(input_seq)  # [1, T_s]
        src_seq = src_seq.repeat(topk, axis=0)  # [1 * k, T_s]
        enc_out = self.encode_model.predict_on_batch(src_seq)  # [1 * k, T_s, model_dim]

        tgt_tokenizer = self.tgt_tokenizer

        start_token_id = tgt_tokenizer.word_index[tgt_tokenizer.start_token]
        end_token_id = tgt_tokenizer.word_index[tgt_tokenizer.end_token]

        target_seq = np.zeros((topk, self.tgt_max_len))  # [1 * k, T_t]
        target_seq[:, 0] = start_token_id

        sequences = [([], 0.0)]
        final_results = []

        for i in range(self.tgt_max_len - 1):
            if len(final_results) >= topk: break
            output = self.decode_model.predict_on_batch([src_seq, enc_out, target_seq])  # [1 * k, T_t, model_dim]
            k_cur_output = output[:, i, :]  # [1 * k, model_dim]

            all_candidates = []

            for k, cur_output in zip(range(len(sequences)), k_cur_output):
                seq, score = sequences[k]

                # Find a complete sentence, add to the final result.
                if target_seq[k, i] == end_token_id:
                    final_results.append((seq[:-1], score))
                    continue

                # Other sentences will be generated among the remaining candidates.
                wsorted = sorted(list(enumerate(cur_output)), key=lambda x: x[-1], reverse=True)
                for wid, wp in wsorted[:topk]:
                    all_candidates.append((seq + [wid], score + wp))

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

            sequences = ordered[:topk]
            for kk, cc in enumerate(sequences):
                seq, score = cc
                target_seq[kk, 1: len(seq) + 1] = seq

        # Extend if last word is not end_token.
        final_results.extend(sequences)
        final_results = [(x, y / (len(x) + 1)) for x, y in final_results]
        final_results = sorted(final_results, key=lambda tup: tup[1], reverse=True)[:topk]

        ori_split = tgt_tokenizer.split
        tgt_tokenizer.split = delimiter
        sequences = [(tgt_tokenizer.sequences_to_texts([x])[0], y) for x, y in final_results]
        tgt_tokenizer.split = ori_split
        return sequences

    def beam_search_text_decode(self, texts,
                                k=5, delimiter=' '):
        assert self.src_tokenizer is not None
        sequences = self.src_tokenizer.texts_to_sequences(texts)
        return self.beam_search_sequence_decode(sequences, k, delimiter)

    @staticmethod
    def get_or_create(config, optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-6),
                      src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None):

        if Transformer.__singleton is None:
            if type(config) == dict:
                config = config
            elif type(config) == str:
                with open(config, mode='r') as file:
                    config = dict(json.load(file))
            config['optimizer'] = optimizer
            if src_dict_path is not None:
                config['src_tokenizer'] = load_dictionary(src_dict_path)
            if tgt_dict_path is not None:
                config['tgt_tokenizer'] = load_dictionary(tgt_dict_path)
            Transformer.__singleton = Transformer(**config)
            try:
                Transformer.__singleton.model.load_weights(weights_path)
            except:
                print("\nNo weights found, create a new model.")
        return Transformer.__singleton
