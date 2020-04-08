from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def create_tokenizer(num_words, texts):
  tokenizer = Tokenizer(num_words=num_words)
  tokenizer.fit_on_texts(texts)
  return tokenizer.texts_to_sequences, tokenizer.sequences_to_texts, tokenizer