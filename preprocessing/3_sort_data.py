import json

# Hàm lọc nhãn chỉ lấy sample có nhãn của khách hàng nằm trong bộ dữ liệu nhãn
def filter_customer_data(data_file, label_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        document_data = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    match_data = []
    labels = [item["Nghiệp vụ chi tiết (đầu ra)"] for item in label_data]

    for i, doc_item in enumerate(document_data):
        doc_type = doc_item["Loại CT"]
        describe_doc = doc_item["Diễn giải"]
        sample_label = doc_item["Loại nghiệp vụ"]

        for j, label_item in enumerate(label_data):
            label_type = label_item["Loại chứng từ chuẩn hoá"]
            describe_label = label_item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""]
            label_name = label_item["Nghiệp vụ chi tiết (đầu ra)"]

            if doc_type == label_type and sample_label in label_name:
                match_data.append(doc_item)
                break

    with open("match_data.json", 'w', encoding='utf-8') as f:
        json.dump(match_data, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu các mục khớp vào match_customers.json")

# thống kê chứng từ không có trong dữ liệu nhãn
def check_missing_chung_tu(data_file, label_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        document_data = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    missing_documents = []

    # Lấy danh sách các "Loại chứng từ chuẩn hoá" từ dữ liệu nhãn
    label_types = [label_item.get("Loại chứng từ chuẩn hoá") for label_item in label_data]

    for doc_item in document_data:
        doc_type = doc_item.get("Loại CT")
        if doc_type not in label_types:
            missing_documents.append(doc_item)

    with open('missing_documents.json', 'w', encoding='utf-8') as f:
        json.dump(missing_documents, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu các chứng từ không có trong nhãn vào missing_documents.json")

# lọc dữ liệu text trùng trong dữ liệu khách hàng
def remove_duplicate_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        document_data = json.load(f)

    unique_entries = []
    seen_entries = set()

    for doc_item in document_data:
        doc_type = doc_item.get("Loại CT")
        describe = doc_item.get("Diễn giải")

        entry_key = (doc_type, describe)

        if entry_key not in seen_entries:
            seen_entries.add(entry_key)
            unique_entries.append(doc_item)

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(unique_entries, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu dữ liệu sau khi loại bỏ các bản ghi trùng vào filtered_data.json")


# lọc dữ liệu text gần giông nhau trong dữ liệu khách hàng (có cùng chứng từ và nhãn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def filter_data_same_same(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        customer_data = json.load(f)

    document_type = [item["Loại CT"] for item in customer_data]
    labels = [item["Loại nghiệp vụ"] for item in customer_data]
    descriptions = [item["Diễn giải"] for item in customer_data]

    # Biểu diễn "Diễn giải" thành vector bằng TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    threshold = 0.7

    keep_indices = []
    removed_indices = []

    removed = set()

    for i in range(len(descriptions)):
        if i in removed:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(descriptions)):
            if document_type[i] == document_type[j] and labels[i] == labels[j]:
                if similarity_matrix[i, j] > threshold:
                    removed.add(j)
                    removed_indices.append(j)

    filtered_data = [customer_data[i] for i in keep_indices]
    removed_data = [customer_data[i] for i in removed_indices]

    print(f"Số lượng mẫu ban đầu: {len(customer_data)}")
    print(f"Số lượng mẫu sau khi lọc: {len(filtered_data)}")
    print(f"Số lượng mẫu bị loại bỏ: {len(removed_data)}")

    with open('filtered_customer_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    with open('removed_customer_data.json', 'w', encoding='utf-8') as f:
        json.dump(removed_data, f, ensure_ascii=False, indent=4)

    print("Dữ liệu đã được lưu thành công.")


if __name__ == "__main__":    
    data = "complete_dataset.json"
    label = "Danh sách loại nghiệp vụ1.json"
    # remove_duplicate_data(data)
    filter_data_same_same(data)


