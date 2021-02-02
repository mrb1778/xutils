from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

import tensorflow_datasets as tfds


def tokenize(texts, vocab_size=1000, oov_token="<OOV>", max_length=None):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    if max_length is not None:
        return pad_sequences(sequences, truncating='post', maxlen=max_length), tokenizer
    else:
        return sequences, tokenizer



# keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
#                                            truncating='post', maxlen=max_len)

def create_embedding_layer(vocab_size=1000, embedding_dim=16, max_length=20):
    return Embedding(vocab_size, embedding_dim, input_length=max_length)
