# (phương pháp mới) so sánh dữ liệu với nhãn có cùng chứng từ

import json
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
torch.cuda.empty_cache()

class EmbeddingSimilarityLabeler:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.label_embeddings = {}

# add nhãn và mô tả vào embedding (có thể mở rộng)
    def add_label(self, document_type, label, description):
        inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if document_type not in self.label_embeddings:
            self.label_embeddings[document_type] = {}
        self.label_embeddings[document_type][label] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def predict(self, text, document_type):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        text_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        if document_type in self.label_embeddings:
            similarities = {label: cosine_similarity(text_embedding, emb)[0][0]
                            for label, emb in self.label_embeddings[document_type].items()}

            top_label = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:1]
            return [top_label]

customer_file = '/content/test_ko_train_loctrung.json'
label_file = '/content/Danh sách loại nghiệp vụ1.json'

with open(customer_file, 'r', encoding='utf-8') as f:
    customer_data = json.load(f)

with open(label_file, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

labeler = EmbeddingSimilarityLabeler("/content/bkai-embedding-encoder")

# Đưa nhãn và mô tả vào embedding
for label_item in label_data:
    document_type = label_item["Loại chứng từ chuẩn hoá"].strip()
    labels = label_item["Nghiệp vụ chi tiết (đầu ra)"].strip().split('; ')
    for label in labels:
        label_desc = label_item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""].strip()
        labeler.add_label(document_type, label, label_desc)

# Dự đoán nhãn cho nội dung "Diễn giải" dựa trên nhãn đã đưa vào không gian embedding
for i, customer_item in enumerate(customer_data):
    customer_type = customer_item["Loại CT"].strip()
    customer_text = customer_item["Diễn giải"].strip()
    if customer_text is None or customer_text.strip() == "":
        continue

    top_label = labeler.predict(customer_text, customer_type)
    top_label = top_label[0]
    if top_label:
        customer_item['label top 1'] = top_label[0][0]
        print(f"Best label for document {i+1}: {top_label}")
    else:
        customer_item['label top 1'] = "Không có nhãn phù hợp"
        print(f"No matching label found for document {i+1}")

with open('updated_customer_data.json', 'w', encoding='utf-8') as f:
    json.dump(customer_data, f, ensure_ascii=False, indent=4)