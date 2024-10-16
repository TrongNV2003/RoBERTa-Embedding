# tính accuracy bằng cách so sánh 2 labels
import json

def calculate_accuracy(data_file, label_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    total_records = 0
    total_score = 0
    max_score = 100

    for item in data:
        customer_label = item.get("Loại nghiệp vụ").strip()
        first_predict_label = item.get("label top 1").strip()
        describe = item.get("Diễn giải").strip()
        for label_item in label_data:
            label_type = label_item["Loại chứng từ chuẩn hoá"].strip()
            labels = label_item["Nghiệp vụ chi tiết (đầu ra)"].strip()
            label_type = label_item["Loại chứng từ"].strip()

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

        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

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
    data_file = "/content/updated_customer_data.json"
    label_file = "/content/Danh sách loại nghiệp vụ1.json"
    calculate_accuracy(data_file, label_file)