import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Step 1: Create a simple sentiment classification dataset
data = {
    'sentence': [
        'I love this product!',
        'This is the best movie I have ever seen.',
        'Absolutely wonderful experience.',
        'I hate this item.',
        'This is the worst movie ever.',
        'Terrible experience, will not recommend.',
        'Amazing service and friendly staff.',
        'I will never buy this again.',
        'Highly recommend to everyone!',
        'The quality is very poor.'
    ],
    'label': [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Tokenize the dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label)

tokenizer = AutoTokenizer.from_pretrained('../huggingface_models/bert-base-cased')
train_texts, val_texts, train_labels, val_labels = train_test_split(df['sentence'].tolist(), df['label'].tolist(), test_size=0.2)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Step 3: Define the model, optimizer, and scheduler
model = AutoModelForSequenceClassification.from_pretrained('../huggingface_models/bert-base-cased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 6
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Step 4: Define training and validation functions
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def val_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(predictions == labels)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return avg_loss, accuracy

# Step 5: Train and evaluate the model
num_epochs = 6
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    val_loss, val_accuracy = val_epoch(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train loss: {train_loss}')
    print(f'Validation loss: {val_loss}')
    print(f'Validation accuracy: {val_accuracy}')

# Example prediction
model.eval()
inputs = tokenizer("I really enjoy this!", return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = F.softmax(logits, dim=-1)
predicted_class_id = logits.argmax().item()
print(f"Predicted class ID: {predicted_class_id}")
print(f"Prediction probabilities: {probabilities.cpu().numpy()}")