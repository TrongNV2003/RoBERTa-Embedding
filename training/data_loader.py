import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple

class QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer) -> None:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["Diễn giải"]
        # label_text = item["Loại nghiệp vụ"]
        describe = item["Mô tả chi tiết cho \"Nghiệp vụ chi tiết\""]

        # Encode context and describe to input_ids and attention_mask
        input_ids = self.tokenizer.encode(
            f"{context} <sep> {describe}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    #     encoded_text = self.tokenizer(
    #         text,
    #         padding="max_length",
    #         max_length=self.max_length,
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #     return (
    #         encoded_text["input_ids"].squeeze(0),
    #         encoded_text["attention_mask"].squeeze(0)
    #     )
