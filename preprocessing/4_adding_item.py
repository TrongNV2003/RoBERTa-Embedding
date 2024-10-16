import json

# lấy cả item
def get_item(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    filtered_data = [item for item in data if item.get("Loại CT") == "Thu tiền gửi"]

    # Ghi các object lọc được vào file mới
    with open('extract_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Đã lấy {len(filtered_data)} object.")

# xoá cả item
def delete_object(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    filtered_data = [item for item in data if item.get("Loại CT") != "Thu tiền gửi"]

    # Ghi các object lọc được vào file mới
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Đã xoá object trong {json_file}.")

# xoá 1 trường trong item
def remove_item(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if "Diễn giải chi tiết" in item:
            del item["Diễn giải chi tiết"]

    # Lưu lại dữ liệu đã sửa đổi vào file JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Đã loại bỏ trường 'Nội dung đào tạo AVA' thành công.")


# gán dữ liệu mô tả nhãn vào từng dữ liệu customers
def match_and_add_description(labels_file, customers_file):
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    with open(customers_file, 'r', encoding='utf-8') as f:
        customers_data = json.load(f)

    for customer in customers_data:
        customer_ct_type = customer.get("Loại CT").strip()
        customer_business_type = customer.get("Loại nghiệp vụ").strip()

        for label in labels_data:
            label_ct_type = label.get("Loại chứng từ chuẩn hoá").strip()
            label_business_type = label.get("Nghiệp vụ chi tiết (đầu ra)").strip()
            
            if customer_ct_type == label_ct_type and customer_business_type == label_business_type:
                customer["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""] = label.get("Mô tả chi tiết cho \"Nghiệp vụ chi tiết\"")
                break

    with open(customers_file, 'w', encoding='utf-8') as f:
        json.dump(customers_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lưu vào file {customers_file}.")


# add item to sample
def add_item(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        item['label'] = 1

    with open (json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("Adding label successfully")


# create negative sample
import random
import copy

def negative_sample(json_file):
    # Mở file JSON để load dữ liệu ban đầu
    with open(json_file, 'r', encoding='utf-8') as f:
        customer_data = json.load(f)

    # Gán label cho positive samples
    positive_data = []
    for item in customer_data:
        item_with_label = copy.deepcopy(item)
        item_with_label['label'] = 1  # Label = 1 cho các mẫu đúng
        positive_data.append(item_with_label)

    # Tạo một danh sách các negative samples
    negative_data = []

    # Tạo một bản sao của customer_data để hoán đổi nhãn
    for i, customer_item in enumerate(customer_data):
        neg_sample = copy.deepcopy(customer_item)

        # Tiếp tục hoán đổi cho đến khi giá trị của cả "Loại nghiệp vụ" và "Mô tả chi tiết cho 'Nghiệp vụ chi tiết'" thay đổi
        while True:
            random_index = random.choice([x for x in range(len(customer_data)) if x != i])
            
            # Lấy giá trị hoán đổi từ một mẫu khác
            new_loai_nghiep_vu = customer_data[random_index]["Loại nghiệp vụ"]
            new_mo_ta_chi_tiet = customer_data[random_index]["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""]
            
            # Kiểm tra xem giá trị mới có khác với giá trị ban đầu không
            if (new_loai_nghiep_vu != neg_sample["Loại nghiệp vụ"] or
                new_mo_ta_chi_tiet != neg_sample["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""]):
                
                # Thay đổi cả cặp "Loại nghiệp vụ" và "Mô tả chi tiết cho 'Nghiệp vụ chi tiết'"
                neg_sample["Loại nghiệp vụ"] = new_loai_nghiep_vu
                neg_sample["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""] = new_mo_ta_chi_tiet
                break  # Thoát vòng lặp khi đã hoán đổi thành công

        # Gán label = 0 cho các mẫu negative
        neg_sample['label'] = 0
        negative_data.append(neg_sample)

    # Kết hợp cả positive và negative samples vào một tập dữ liệu duy nhất
    combined_data = positive_data + negative_data

    # Lưu lại dữ liệu đã kết hợp vào file JSON mới
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lưu thành công vào {json_file}.")


if __name__ == "__main__":
    label_file = "Danh sách loại nghiệp vụ1.json"
    data_file = "dataset/test_intrain.json"
    negative_data = "negative sample/thu_tien_gui.json"
    
    # get_item(fake_data)
    # delete_object(fake_data)
    add_item(data_file)

