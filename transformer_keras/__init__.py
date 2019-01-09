import json

from transformer_keras.core import Transformer

get_or_create = Transformer.get_or_create


def save_config(transformer, config_path):
    with open(config_path, mode="w+") as file:
        json.dump(transformer.get_config(), file)
