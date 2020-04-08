from tensorflow.keras.layers import Embedding, Input, GRU, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from .callbacks import create_callbacks

def create_nmt_model(num_words):
	# Encoder
	# Input layer
	encoder_input = Input(shape=(None,), name="encoder_input")

	# Embedding layer
	embedding_size=128
	encoder_embedding = Embedding(input_dim=num_words,
								output_dim=embedding_size,
								name="encoder_embedding")
	# Encoder's hidden state size
	encoder_hidden_state_size = 512
	encoder_gru1 = GRU(encoder_hidden_state_size, name="encoder_gru1", return_sequences=True)
	encoder_gru2 = GRU(encoder_hidden_state_size, name="encoder_gru2", return_sequences=True)
	encoder_gru3 = GRU(encoder_hidden_state_size, name="encoder_gru3", return_sequences=False)

	def create_encoder():
		# Embeddings
		net = encoder_embedding(encoder_input)
		# Neural net
		net = encoder_gru1(net)
		net = encoder_gru2(net)
		net = encoder_gru3(net)
		# Output
		return net

	# Decoder
	# Input layer
	decoder_input = Input(shape=(None,), name="decoder_input")

	# Initial state (context state)
	decoder_initial_state = Input(shape=(encoder_hidden_state_size,),
								name='decoder_initial_state')

	# Embedding layer
	embedding_size=128 # TODO: We might need to lower this since the vocabulary is very small
	decoder_embedding = Embedding(input_dim=num_words,
								output_dim=embedding_size,
								name="decoder_embedding")
	# Decoder's hidden state size
	decoder_hidden_state_size = 512
	decoder_gru1 = GRU(decoder_hidden_state_size, name="decoder_gru1", return_sequences=True)
	decoder_gru2 = GRU(decoder_hidden_state_size, name="decoder_gru2", return_sequences=True)
	decoder_gru3 = GRU(decoder_hidden_state_size, name="decoder_gru3", return_sequences=True)
	
	# Decoder dense layer
	decoder_dense = Dense(num_words, activation="linear", name="decoder_dense")

	def create_decoder(initial_state):
		# Embedding layer
		net = decoder_embedding(decoder_input)
		# Connect all the GRU layers
		net = decoder_gru1(net, initial_state=initial_state)
		net = decoder_gru2(net, initial_state=initial_state)
		net = decoder_gru3(net, initial_state=initial_state)

		# The dense layer
		net = decoder_dense(net)
		return net


	# Creating the outputs for each of the encoder and the decoder
	encoder_output = create_encoder()
	decoder_output = create_decoder(initial_state=encoder_output)
	model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

	# Optimizer
	optimizer = Adam(learning_rate=0.003)

	# Loss function
	loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
	def loss_function(real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	model_train.compile(optimizer=optimizer,
                    loss=loss_function)

	print(model_train.summary())

	callbacks = create_callbacks()

	def train(encoder_input_data,
				decoder_input_data,
				decoder_output_data,
				epochs=10,
				validation_split=0.3):
		history = model_train.fit(x={"encoder_input":encoder_input_data, "decoder_input": decoder_input_data},
					y={"decoder_dense": decoder_output_data},
					batch_size=512,
					epochs=epochs,
					validation_split=validation_split,
					callbacks=callbacks)

	# Create the encoder
	model_encoder = Model(inputs=[encoder_input],
						outputs=[encoder_output])
	# Create the decoder
	decoder_output = create_decoder(initial_state=decoder_initial_state)
	model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
						outputs=[decoder_output])

	def predict(texts):
		input_tokens = preprocess_input(texts)
		decoder_initial_state_data = model_encoder.predict(input_tokens)

		# Since the decoder input will depend on the output,
		# we will store it in this vector
		decoder_input_data = np.zeros(shape=(1, chosen_max_tokens_length_decoder), dtype=np.int)

		token_start = output_tokenize(["ssss"])[0][0]
		token_end = output_tokenize(["eeee"])[0][0]
		current_token = token_start
		count_tokens = 0
		# Output text
		output_text= ""

		while current_token != token_end and count_tokens < chosen_max_tokens_length_decoder:
			decoder_input_data[0, count_tokens] = current_token
			decoder_data = {
				"decoder_initial_state": decoder_initial_state_data,
				"decoder_input": decoder_input_data
			}
			decoder_output = model_decoder.predict(decoder_data)
			token_onehot = decoder_output[0, count_tokens, :]
			current_token = np.argmax(token_onehot)
			output_text += " " + output_reverse_tokenize([[current_token]])[0]
			count_tokens +=1

		return output_text

	return train, predict