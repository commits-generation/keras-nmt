from sklearn.model_selection import train_test_split
from app.preprocess.tokenizer import create_tokenizer
import numpy as np

def run(input_series, 
		output_series, 
		max_input_length=118,
		max_input_length=28,
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

	# Create tokenizer for the input and the output
	input_tokenize, input_reverse_tokenize, _input_tokenizer = create_tokenizer(
		num_words=number_of_words,
		texts=np.array(x_train))
	output_tokenize, output_reverse_tokenize, _output_tokenizer = create_tokenizer(
		num_words=number_of_words,
		texts=y_train)

	# Tokenize the input and the output
	x_tokens = input_tokenize(x_train)
	y_tokens = output_tokenize(y_train)

	
	

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