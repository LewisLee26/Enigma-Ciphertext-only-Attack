import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_from_disk
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import wandb

wandb.init(project="Enigma Ciphertext-only Attack")


# Define a character-level tokenizer
class CharTokenizer:
    def __init__(self):
        pass

    def __call__(self, text):
        return list(text)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained DistilBERT model and tokenizer for sequence classification
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)


# This model uses a mix of character-level tokenizer and the default DistilBERT tokenizer
char_tokenizer = CharTokenizer()
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the dataset
def loadDataset(path, batch_size, data_cutoff, shuffle):
    dataset = load_from_disk(path)

    # Sample data for binary classification
    texts = dataset['text'][:data_cutoff]
    labels = dataset['label'][:data_cutoff]

    # Tokenize input texts at the character level
    tokenized_texts = [char_tokenizer(text) for text in texts]

    # Convert characters to token ids using the DistilBERT tokenizer
    input_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(tokens))) for tokens in tokenized_texts]

    # Pad sequences to the same length
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]

    # Convert to PyTorch tensor
    input_ids = torch.tensor(padded_input_ids).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    labels = torch.tensor(labels).to(device)

    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


batch_size = 10

wandb.config.update({"batch_size": batch_size})

train_dataloader = loadDataset(r"dataset\enigma_binary_classification_en_0-13_plugs\train",
                               batch_size=batch_size,
                               data_cutoff=1000,
                               shuffle=True,
                               )

test_dataloader = loadDataset(r"dataset\enigma_binary_classification_en_0-13_plugs\test",
                              batch_size=batch_size,
                              data_cutoff=None,
                              shuffle=False,
                              )


# Train the model
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=2, num_training_steps=num_training_steps
)

train_pbar = tqdm(range(num_training_steps), desc="Training loss: None")

model.train()

for epoch in range(num_epochs):
    running_loss = 0
    for i, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        running_loss += loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_pbar.set_description(f"Epoch: {epoch+1} - Training loss: {running_loss/(i+1):.5f}")
        train_pbar.update(1)

        wandb.log({"training_loss": loss.item()})
        

model.save_pretrained(r"model/enigma_binary_classification_en_0-13_plugs")

# Test the model

all_predictions = []
all_labels = []

model.eval()

test_pbar = tqdm(range(len(test_dataloader)), desc="Testing accuracy: None")
for i, batch in enumerate(test_dataloader):
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)


    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    if i % 100 == 0:
        accuracy = accuracy_score(all_labels, all_predictions)
        test_pbar.set_description(f"Testing accuracy: {accuracy:.5f}")
    

    test_pbar.update(1)


accuracy = accuracy_score(all_labels, all_predictions)

print(f"Accuracy: {accuracy}")

# while True:
#     print()
#     user_input = input("input: ")

#     tokenized_text = char_tokenizer(user_input)

#     # Convert characters to token ids using the DistilBERT tokenizer
#     input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(tokenized_text)))

#     # Pad sequences to the same length
#     len_ids = len(input_ids)
#     padded_input_ids = input_ids + [tokenizer.pad_token_id] * (512 - len_ids)

#     # Convert to PyTorch tensor
#     input_ids = torch.tensor(padded_input_ids).to(device)
#     attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

#     output = model(input_ids, attention_mask=attention_mask)
#     print(output)
#     print(output.logits)

# results = metric.compute()
# print(results)