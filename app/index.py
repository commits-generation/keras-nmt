from sklearn.model_selection import train_test_split
from app.preprocess import create_preprocessor
import numpy as np
from app.model.nmt import create_nmt_model

def run(input_series, 
		output_series, 
		max_input_length=118,
		max_output_length=28,
		number_of_words=50000,
		test_size=.05, 
		validation_size=.2):
	'''
	Run the NMT model

	Arguments:
		- number_of_words: Vocabulary size
	'''
	# Split the dataset to train and test
	x_train, x_test, y_train, y_test = train_test_split(
		input_series, 
		output_series.apply(lambda x: "ssss "+x+" eeee"), test_size=test_size)

	# Create the preprocessors for the input and the output
	preprocess_input, _ = create_preprocessor(num_words=number_of_words,
											texts=x_train,
											max_length=max_input_length)
	preprocess_output, _ = create_preprocessor(num_words=number_of_words,
											texts=y_train,
											max_length=max_output_length)
	x_train_preprocessed = preprocess_input(x_train)
	y_train_preprocessed = preprocess_input(x_train)


	# Create NMT model
	train, predict = create_nmt_model(num_words=number_of_words)

	encoder_input_data= x_train_preprocessed[:]
	decoder_input_data= y_train_preprocessed[:, :-1]
	decoder_output_data= y_train_preprocessed[:, 1:]

	train(encoder_input_data=encoder_input_data,
			decoder_input_data=decoder_input_data,
			decoder_output_data=decoder_output_data,
			epochs=1,)
		

	
	

if __name__ == "__main__":
	from commit_generation_datasets.datasets import load_dataset, tables
	# Load the dataset
	from commit_generation_datasets.datasets import load_dataset, tables
	commits_original_dataset = load_dataset(tables.commits_with_scores_v2)
	refactoring_original_dataset = load_dataset(tables.refactorings_in_commit_v1)[["commitID","refactorings"]]
	# Join the two tables to get the refactorings (as a list) for each commit
	df = commits_original_dataset.merge(refactoring_original_dataset, left_on="CommitID", right_on="commitID", how="left")
	run(df.refactorings, df.CommitMessage)
	pass