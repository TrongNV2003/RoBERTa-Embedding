import json

"""
Xử lý nhãn dữ liệu
"""
# tách thành các chứng từ riêng trong dữ liệu nhãn
def split_document_types(input_file):
    # Hàm con để tách loại chứng từ ra thành các object riêng biệt
    def split_document_types(data):
        new_data = []
        for item in data:
            document_types = item["Loại chứng từ "].replace(";", "\n").split("\n")
            for doc_type in document_types:
                new_item = item.copy()
                new_item["Loại chứng từ "] = doc_type.strip()
                new_data.append(new_item)
        return new_data

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = split_document_types(data)

    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được tách và lưu vào file {input_file}")

# chuấn hoá chứng từ
def standardize_label(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw = [
        {
            "Chuyển khoản kho bạc về tài khoản tiền gửi",
            "Chuyển khoản kho bạc mua VTHH"
        },
        {
            "Phiếu chi nộp tiền vào ngân hàng, kho bạc"
        },
        {
            "Phiếu chi"
        },
        {
            "Phiếu thu bán hàng"
        },
        {
            "Phiếu thu tiền khách hàng"
        },
        {
            "Thu tiền gửi từ bán hàng"
        },
        {
            "Thu tiền gửi từ khách hàng"
        },
        {
            "Chi tiền gửi trả nhà cung cấp"
        },
        {
            "Chuyển khoản thanh toán bảo hiểm"
        },
        {
            "Chứng từ nghiệp vụ khác"
        },
        {
            "Chuyển khoản kho bạc vào tài khoản tiền gửi",
            "Chuyển khoản kho bạc vào TK tiền gửi"
        }
    ]
    
    standardize = [
        "Chuyển khoản kho bạc",
        "Phiếu chi nộp tiền vào ngân hàng, kho bạc",
        "Phiếu chi",
        "Phiếu thu",
        "Phiếu thu tiền khách hàng",
        "Thu tiền gửi",
        "Thu tiền gửi từ khách hàng",
        "Chi tiền gửi",
        "Chuyển khoản thanh toán bảo hiểm",
        "Chứng từ nghiệp vụ khác",
        "Chuyển khoản kho bạc vào tài khoản tiền gửi"
    ]
    
    for item in data:
        raw_data = item.get("Loại chứng từ ")
        if raw_data:
            original_raw_data = raw_data

            standardized_label = None
            for i, raw_group in enumerate(raw):
                for r in raw_group:
                    if r in raw_data:
                        standardized_label = standardize[i] 
                        break 
                if standardized_label:
                    break
            
            if standardized_label:
                raw_data = standardized_label

            if raw_data != original_raw_data or raw_data in standardize:
                item["Loại chứng từ chuẩn hoá"] = raw_data

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Dữ liệu đã được chuẩn hoá và lưu lại trong file {data_file}.")

# đánh STT cho dữ liệu nhãn
def numerical_label(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    for index, item in enumerate(label_data, start=1):
        # item["label"] = item["Nghiệp vụ chi tiết (đầu ra)"]
        item["STT"] = index

    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=4)

    print(f'STT đã được cập nhật lại trong {label_file}.')


"""
Xử lý document dữ liệu
"""

# xử lý từ viết tắt trong "Diễn giải" (text)
def dictionary_standardize_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # with open(label_file, 'r', encoding='utf-8') as f:
    #     label = json.load(f)

    abbreviation_dict = {
        "BHYT": "bảo hiểm y tế",
        "BHXH": "bảo hiểm xã hội",
        "BHTN": "bảo hiểm thất nghiệp",
        "BHTNLD": "bảo hiểm tai nạn lao động",
        "BHTNLĐ": "bảo hiểm tai nạn lao động",
        "KPCĐ": "kinh phí công đoàn",
        "KSK": "khám sức khoẻ",
        "KCB": "khám chữa bệnh",
        "KPCD": "kinh phí công đoàn",
        "GTGT": "giá trị gia tăng",
        "TNCN": "thu nhập cá nhân",
        "TNDN": "thu nhập doanh nghiệp",
        " VPP ": " văn phòng phẩm ",
        " CBGV ": " cán bộ giáo viên ",
        " GVTD ": " giáo viên thể dục ",
        "HĐLĐ": "hợp đồng lao động",
        "NSNN": "ngân sách nhà nước",
        "ĐPCĐ": "đoàn phí công đoàn",
        "bhyt": "bảo hiểm y tế",
        " HP ": " học phí ",
        "NCS": "nghiên cứu sinh",
        "VKS": "Viện kiểm sát",
        " TK ": " tài khoản ",
        " PC ": " phụ cấp ",
        " TT ": " thanh toán ",
        "UBKT": "ủy ban kiểm tra",
        " đ/c ": " đồng chí ",
        "LLCT": "lý luận chính trị",
        " KB ": " kho bạc ",
        "Bảo hiểm XH": "bảo hiểm xã hội",
        "CNTT": "công nghệ thông tin",
        "GTVT": "giao thông vận tải",
    }

    for item in data:
        description = item.get("Diễn giải")
        if isinstance(description, str):
            for abbr, standard in abbreviation_dict.items():
                description = description.replace(abbr, standard)
            item["Diễn giải"] = description

    # for item in label:
    #     description = item.get("Mô tả chi tiết cho \"Nghiệp vụ chi tiết\"")
    #     if isinstance(description, str):
    #         for abbr, standard in abbreviation_dict.items():
    #             description = description.replace(abbr, standard)
    #         item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""] = description

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    # with open(label_file, 'w', encoding='utf-8') as f:
    #     json.dump(label, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được chuẩn hóa và lưu vào file {data_file}.")


# lọc dữ liệu null ở cột "Loại nghiệp vụ" và "Diễn giải"
def remove_null_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    # new_data = [item for item in data if item.get("Loại nghiệp vụ") is not None]
    
    for item in data:
        label = item.get("Loại nghiệp vụ")
        text = item.get("Diễn giải")
        if label is not None and text is not None:
            new_data.append(item)

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lọc và lưu vào file {data_file}.")
    
if __name__ == "__main__":
    data_file = "json data/Viện công nghệ thông tin (1).json"
    dictionary_standardize_data(data_file)

