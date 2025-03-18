# Libraries
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face components
import torch
from torch.optim import AdamW # Optimizer for training
from tqdm import tqdm   # Progress bar utilities
from helpers import set_seed, load_and_prepare_data, calculate_accuracy # Helper functions created locally

def get_hyperparameters():
    """
    Returns training hyperparameters.

    Returns:
        tuple: (num_epochs, batch_size, learning_rate)
    """

    # Train for fewer epochs as sequence classification converges faster
    num_epochs = 8

    # Standard batch size that works well with most GPU memory
    batch_size = 16

    # Standard learning rate for fine-tuning transformers
    learning_rate = 5e-5

    return num_epochs, batch_size, learning_rate

if __name__ == '__main__':

    # Set random seed
    set_seed(25)

    # Location of dataset
    # Source: https://huggingface.co/datasets/hungnm/multilingual-amazon-review-sentiment-processed
    train_data_dir = '../data/train.hf'
    valid_data_dir = '../data/valid.hf'
    test_data_dir = '../data/test.hf'

    # Select model
    model_name = "openai-community/gpt2"

    # Get hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Setup device: CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

     # Prepare data and get label mappings
    train_loader, valid_loader, test_loader, label_to_id, id_to_label, unique_labels = load_and_prepare_data(
        train_data_dir,
        valid_data_dir,
        test_data_dir,
        tokenizer,
        batch_size
    )

    # Initialize model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels)
    ).to(device)

    # Configure model's label handling
    model.config.pad_token_id = model.config.eos_token_id
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Backward pass and optimization
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Display epoch metrics
        avg_loss = total_loss / num_batches
        test_acc = calculate_accuracy(model, test_loader)
        print(f"Average loss: {avg_loss:.4f}, test accuracy: {test_acc:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("../app/finetuned_model")
    tokenizer.save_pretrained("../app/finetuned_model")