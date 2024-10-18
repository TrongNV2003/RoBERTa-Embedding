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


# ----------------------------------------------
MODEL = "roberta-embeddings-contrastive-learning-full-dataset"
labeler = EmbeddingSimilarityLabeler(MODEL)

def add_label_file(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    for label_item in label_data:
        document_type = label_item["Loại chứng từ chuẩn hoá"].strip()
        labels = label_item["Nghiệp vụ chi tiết (đầu ra)"].strip().split('; ')
        for label in labels:
            label_desc = label_item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""].strip()
            labeler.add_label(document_type, label, label_desc)


def predict_label(customer_file):
    with open(customer_file, 'r', encoding='utf-8') as f:
        customer_data = json.load(f)
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
    

def calculate_accuracy(data_file, label_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    total_records = 0
    total_score = 0
    max_score = 10

    for item in data:
        customer_label = item.get("Loại nghiệp vụ").strip()
        first_predict_label = item.get("label top 1").strip()
        describe = item.get("Diễn giải").strip()
        
        for label_item in label_data:
            label_type = label_item["Loại chứng từ chuẩn hoá"].strip()
            labels = label_item["Nghiệp vụ chi tiết (đầu ra)"].strip()

        if customer_label is None or first_predict_label is None or describe is None:
            continue

        total_records += 1

        labels_processed = [
            item.get("label top 1")
            # item.get("label top 2"),
            # item.get("label top 3"),
            # item.get("label top 4"),
            # item.get("label top 5")
        ]

        # weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        match_score = 0
        for idx, label in enumerate(labels_processed):
            if label is not None and customer_label.strip().lower() == first_predict_label.strip().lower():
                # match_score = max_score * weights[idx]
                match_score = max_score
                break

        total_score += match_score

    if total_records > 0:
        accuracy = (total_score / (total_records * max_score)) * 100
    else:
        accuracy = 0.0

    print("Tên file: ", data_file)
    print(f"Tổng số bản ghi hợp lệ: {total_records}")
    print(f"Độ chính xác: {accuracy:.2f}%")

if __name__ == "__main__":
    customer_file = 'dataset/test_intrain.json'
    label_file = 'Danh sách loại nghiệp vụ1.json'
    output_file = "updated_customer_data.json"
    add_label_file(label_file)
    predict_label(customer_file)
    calculate_accuracy(output_file, label_file)