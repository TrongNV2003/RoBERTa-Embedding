import json
from collections import defaultdict


# thống kê các loại nhãn nghiệp vụ theo loại chứng từ
def extract_descriptions_by_doc_type(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Dictionary để lưu các loại chứng từ và các mô tả chi tiết tương ứng
    doc_type_to_descriptions = {}

    for item in data:
        # Lấy giá trị của "Loại chứng từ chuẩn hoá" và "Loại nghiệp vụ"
        doc_type = item.get("Loại CT", None)
        description = item.get("Loại nghiệp vụ", None)

        if doc_type and description:
            # Nếu loại chứng từ chưa có trong từ điển, khởi tạo một danh sách rỗng
            if doc_type not in doc_type_to_descriptions:
                doc_type_to_descriptions[doc_type] = {}
            
            # Nếu mô tả chi tiết chưa có trong từ điển, khởi tạo số lượng là 0
            if description not in doc_type_to_descriptions[doc_type]:
                doc_type_to_descriptions[doc_type][description] = 0
            
            # Tăng số lượng cho loại nghiệp vụ
            doc_type_to_descriptions[doc_type][description] += 1

    # Tạo đầu ra theo định dạng yêu cầu
    output_data = {doc_type: descs for doc_type, descs in doc_type_to_descriptions.items()}

    # Lưu kết quả vào file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lưu vào file {output_file}.")


# thống kê số nhãn được đánh đúng và sai theo loại chứng từ
def statistics_by_document_type(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = defaultdict(lambda: {
        "correct": 0,
        "incorrect": 0,
        "details": defaultdict(lambda: {"correct": 0, "incorrect": 0})
    })
    total_count = 0
    total_correct = 0
    total_incorrect = 0

    for item in data:
        doc_type = item["Loại CT"]
        actual_label = item.get("Loại nghiệp vụ", "")

        # Lấy các nhãn dự đoán từ top 1 đến top 5
        predicted_labels = [
            item.get(f"label top {i+1}", None) for i in range(5)
        ]

        # Kiểm tra điều kiện "Diễn giải" và các nhãn dự đoán
        description = item.get("Diễn giải", "")
        if not description or all(label is None for label in predicted_labels):
            continue

        total_count += 1  

        # Kiểm tra nếu nhãn thực tế có trong các nhãn dự đoán
        if actual_label in predicted_labels:
            stats[doc_type]["correct"] += 1
            total_correct += 1  
            stats[doc_type]["details"][actual_label]["correct"] += 1  # Đúng cho loại nghiệp vụ
        else:
            stats[doc_type]["incorrect"] += 1
            total_incorrect += 1  
            stats[doc_type]["details"][actual_label]["incorrect"] += 1  # Sai cho loại nghiệp vụ

    # Thêm tổng số lượng, tổng đúng và tổng sai vào thống kê
    stats["Total"] = {
        "correct": total_correct,
        "incorrect": total_incorrect,
        "total": total_count
    }

    return stats

def save_statistics_to_file(stats, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu kết quả thống kê vào file {output_file}")

def print_statistics(stats):
    for doc_type, data in stats.items():
        if doc_type == "Total":
            continue
        print(f"Loại chứng từ: {doc_type}")
        print(f"  Số nghiệp vụ đúng: {data['correct']}")
        print(f"  Số nghiệp vụ sai: {data['incorrect']}")
        
        # Tìm loại nghiệp vụ đúng nhiều nhất
        correct_details = data["details"]
        if correct_details:
            max_correct = max(correct_details.items(), key=lambda x: x[1]["correct"])
            print(f"  Nghiệp vụ đúng nhiều nhất: {max_correct[0]} với {max_correct[1]['correct']} lần đúng")

            # Tìm loại nghiệp vụ sai nhiều nhất
            max_incorrect = max(correct_details.items(), key=lambda x: x[1]["incorrect"])
            print(f"  Nghiệp vụ sai nhiều nhất: {max_incorrect[0]} với {max_incorrect[1]['incorrect']} lần sai")
        print()


if __name__ == "__main__":
    # data_file = "Danh sách tìm kiếm chứng từ (3.1) (bkai-labeled).json"
    # output_file = "thong_ke_nghiep_vu.json"

    # statistics = statistics_by_document_type(data_file)

    # # Lưu kết quả vào file JSON
    # save_statistics_to_file(statistics, output_file)

    # # In kết quả thống kê ra màn hình
    # print_statistics(statistics)
    json_file = "dataset/test_in_train.json"
    output_file = "count_label.json"
    
    extract_descriptions_by_doc_type(json_file, output_file)