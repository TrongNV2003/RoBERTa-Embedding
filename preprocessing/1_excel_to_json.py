import pandas as pd
import json

def excel_to_json(excel_file, json_output):
    df = pd.read_excel(excel_file)

    json_data = df.to_json(orient='records', force_ascii=False, indent=4)

    with open(json_output, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
    print("done")

def json_to_excel(excel_output, json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_excel(excel_output, index=False)
    print("done")

def json_to_csv(json_file, csv_output):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False)
    print("done")

if __name__ == '__main__':
    name = "Viện công nghệ thông tin (1)"
    excel_file = f"raw_data/dữ liệu mới/{name}.xls"
    json_file = f"json data/{name}.json"

    excel_to_json(excel_file, json_file)
    