import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, AutoTokenizer
import json
import torch.nn.functional as F
import time
import datetime

class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_encoding = self.tokenizer(
            item["Diễn giải"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label_encoding = self.tokenizer(
            item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "text_input_ids": text_encoding["input_ids"].squeeze(),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(),
            "label_input_ids": label_encoding["input_ids"].squeeze(),
            "label_attention_mask": label_encoding["attention_mask"].squeeze(),
            "label": torch.tensor(item["class"], dtype=torch.float)
        }

# def contrastive_loss(text_embeddings, label_embeddings, temperature=0.06):
#     similarity_matrix = torch.matmul(text_embeddings, label_embeddings.t()) / temperature
#     labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
#     return torch.nn.CrossEntropyLoss()(similarity_matrix, labels)

def cosine_similarity_loss(text_embeddings, label_embeddings, labels, margin=0.5):
    cosine_sim = F.cosine_similarity(text_embeddings, label_embeddings)

    # Tính BCE loss với nhãn 1 cho positive và 0 cho negative
    loss = F.binary_cross_entropy_with_logits(cosine_sim, labels)

    return loss

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(model, train_loader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        for batch in train_loader:
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            label_input_ids = batch["label_input_ids"].to(device)
            label_attention_mask = batch["label_attention_mask"].to(device)
            labels = batch["label"].to(device)

            text_embeddings = model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
            label_embeddings = model(input_ids=label_input_ids, attention_mask=label_attention_mask).last_hidden_state[:, 0, :]

            loss = cosine_similarity_loss(text_embeddings, label_embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            elapsed = format_time(time.time() - t0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Elapsed: {elapsed}.")

# Tải và tiền xử lý dữ liệu
with open("/content/fake_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = ContrastiveDataset(data, tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = RobertaModel.from_pretrained(MODEL)
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  weight_decay = 0.01)
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

train(model, train_loader, optimizer, device, epochs)


saved_model_dir = "./bkai-embedding-encoder/"

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Save model to the saved_model_dir
model.save_pretrained(saved_model_dir)
tokenizer.save_pretrained(saved_model_dir)