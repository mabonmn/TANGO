from datasets import load_dataset
ds = load_dataset("sentence-transformers/wikipedia-en-sentences")
dataset_train = ds['train']
dataset_train.to_csv('wikipedia-en-sentences.csv', index=False)