from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def create_callbacks():
	# Checkpoint callback to save the model
	checkpoint_dir = '/content/drive/My Drive/Temp/PFA_training/training_checkpoint.keras'
	checkpoint_callback = ModelCheckpoint(checkpoint_dir,
								verbose=1,
								save_weights_only=True,
								save_best_only=True)

	# Early stopping callback
	early_stopping_callback = EarlyStopping(patience=3, verbose=1)

	# Tensorboard callback
	tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

	# Callbacks
	callbacks = [checkpoint_callback,
				early_stopping_callback,
				tensorboard_callback]

	return callbacks