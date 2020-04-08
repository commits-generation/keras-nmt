from .tokenizer import create_tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_preprocessor(num_words, texts, max_length):
	# Create tokenizer
	tokenize, reverse_tokenize, _tokenizer = create_tokenizer(
		num_words=num_words,
		texts=np.array(texts))
	
	def preprocess(series):
		tokenized_texts = tokenize(texts)
		padded_texts = pad_sequences(tokenized_texts, 
									maxlen=max_length,
									padding="post",
									truncating="pre")
		return padded_texts

	return preprocess, _tokenizer
