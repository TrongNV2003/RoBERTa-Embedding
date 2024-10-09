import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple

class QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer, label2id: dict) -> None:
        """
        Args:
            json_file (str): Đường dẫn tới file JSON chứa dữ liệu.
            max_length (int): Độ dài tối đa của input.
            pad_mask_id (int): Giá trị padding để mask trong loss.
            tokenizer (AutoTokenizer): Tokenizer dùng để mã hóa text.
            label2id (dict): Bản đồ giữa các loại nghiệp vụ và id tương ứng.
        """
        # Đọc dữ liệu từ file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["Diễn giải"]
        label_text = item["Loại nghiệp vụ"]

        input_ids, attention_mask = self._encode_text(context)

        label = torch.tensor(self.label2id[label_text])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mã hóa text thành input_ids và attention_mask.
        """
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return (
            encoded_text["input_ids"].squeeze(),
            encoded_text["attention_mask"].squeeze()
        )
