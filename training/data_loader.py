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
        context = item["text"]
        describe = item["label_description"]

        text_encoding = self.tokenizer(
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label_encoding = self.tokenizer(
            describe,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(),
            'label_input_ids': label_encoding['input_ids'].squeeze(),
            'label_attention_mask': label_encoding['attention_mask'].squeeze(),
            'label': item['label']
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
    #         encoded_text["input_ids"].squeeze(),
    #         encoded_text["attention_mask"].squeeze()
    #     )
