import random           # For setting seeds and shuffling data
import torch
from torch.utils.data import DataLoader  # For dataset handling
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face components


def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.

    Args:
        seed (int): Seed value for random number generation
    """

    # Set Python's built-in random seed
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set seed for all available GPUs
    torch.cuda.manual_seed_all(seed)
    # Request cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner for consistent behavior
    torch.backends.cudnn.benchmark = False

def create_label_mappings():
    """
    Creates mappings for label IDs.

    Returns:
        tuple: (label_to_id, id_to_label, unique_labels)
    """

    # Define sentiment labels
    unique_labels = ['NEGATIVE', 'POSITIVE'] # Order dependent: label 0 corresponds to negative and label 1 corresponds to positive

    # Create mappings between labels and IDs
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    return label_to_id, id_to_label, unique_labels

def encode_text(tokenizer, text, return_tensor=False):
    """
    Encodes text using the provided tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer
        text (str): Text to encode
        return_tensor (bool): Whether to return PyTorch tensor

    Returns:
        List or tensor of token IDs
    """
    
    # If tensor output is requested, encode with PyTorch tensors
    if return_tensor:
        return tokenizer.encode(
            text, add_special_tokens=False, return_tensors='pt'
        )
    # Otherwise return list of token IDs
    else:
        return tokenizer.encode(text, add_special_tokens=False)

class TextClassificationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for text classification.
    Converts text and labels into model-ready format.

    Args:
        data: Hugging Face Dataset
        tokenizer: Hugging Face tokenizer
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return total number of examples
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            dict: Contains input_ids and labels
        """

        # Get example from dataset
        item = self.data[idx]

        # Convert text to token IDs
        input_ids = encode_text(self.tokenizer, item['text'])

        return {
            "input_ids": input_ids,
            "label": item['label']
        }
    
def collate_fn(batch):
    """
    Collates batch of examples into training-ready format.
    Handles padding and conversion to tensors.

    Args:
        batch: List of examples from Dataset

    Returns:
        dict: Contains input_ids, labels, and attention_mask tensors
    """

    # Find longest sequence for padding
    max_length = max(len(item['input_ids']) for item in batch)

    # Pad input sequences with zeros
    input_ids = [
        item['input_ids'] +
        [0] * (max_length - len(item['input_ids']))
        for item in batch
    ]

    # Create attention masks (1 for tokens, 0 for padding)
    attention_mask = [
        [1] * len(item['input_ids']) +
        [0] * (max_length - len(item['input_ids']))
        for item in batch
    ]

    # Collect labels
    labels = [item['label'] for item in batch]

    # Convert everything to tensors
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask)
    }
    
def load_and_prepare_data(train_data_dir, valid_data_dir, test_data_dir, tokenizer, batch_size):
    """
    Loads and prepares datasets for training.

    Args:
        train_data_dir (str): directory path of training data
        valid_data_dir (str): directory path of validation data
        test_data_dir (str): directory path of test data
        tokenizer: Tokenizer for text processing
        batch_size (int): Batch size for DataLoader

    Returns:
        tuple: (train_dataloader, valid_dataloader, test_dataloader, label_to_id, id_to_label, unique_labels)
    """

    # Load datasets
    train_dataset = load_from_disk(train_data_dir)
    valid_dataset = load_from_disk(valid_data_dir)
    test_dataset = load_from_disk(test_data_dir)

    # Create label mappings
    label_to_id, id_to_label, unique_labels = create_label_mappings()

    # Create datasets
    train_data = TextClassificationDataset(
        train_dataset.select(range(1000)), ##### REMOVE SELECT FILTER TO PROCESS WHOLE DATASET ######
        tokenizer
    )

    valid_data = TextClassificationDataset(
        valid_dataset.select(range(200)), ##### REMOVE SELECT FILTER TO PROCESS WHOLE DATASET ######
        tokenizer
    )

    test_data = TextClassificationDataset(
        test_dataset.select(range(100)), ##### REMOVE SELECT FILTER TO PROCESS WHOLE DATASET ######
        tokenizer
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_dataloader, valid_dataloader, test_dataloader, label_to_id, id_to_label, unique_labels

def calculate_accuracy(model, dataloader):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Fine-tuned model
        dataloader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """

    # Set model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            # Update accuracy counters
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total

    # Reset model to training mode
    model.train()

    return accuracy

def generate_label(model, tokenizer, text):
    """
    Generates label prediction for input text.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        text (str): Input text to classify

    Returns:
        str: Predicted label
    """

    # Encode text and move to model's device
    input_ids = encode_text(
        tokenizer,
        text,
        return_tensor=True
    ).to(model.device)

    # Get model predictions
    outputs = model(input_ids)
    logits = outputs.logits[0]

    # Get class with highest probability
    predicted_class = logits.argmax().item()

    # Convert class ID to label string
    predicted_label = model.config.id2label[predicted_class]

    return predicted_label

def query_model(model_path, query):
    """
    Query a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        query (str): Text to classify
    """

    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate and display prediction
    sentiment = generate_label(model, tokenizer, query)
    
    return sentiment