from datasets import load_dataset

# Load dataset
train_dataset = load_dataset("hungnm/multilingual-amazon-review-sentiment-processed", split="train")
valid_dataset = load_dataset("hungnm/multilingual-amazon-review-sentiment-processed", split="validation")
test_dataset = load_dataset("hungnm/multilingual-amazon-review-sentiment-processed", split="test")

# Save locally
train_dataset.save_to_disk("./data/train.hf")
valid_dataset.save_to_disk("./data/valid.hf")
test_dataset.save_to_disk("./data/test.hf")
