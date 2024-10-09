import argparse
import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from data_loader import QGDataset
from trainer import Trainer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[2e-5])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--qg_model", type=str, default="bkai-foundation-models/vietnamese-bi-encoder")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./bkai-encoder-labeling")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="result/training.csv")
    parser.add_argument("--train_file", type=str, default="dataset/train.json")
    parser.add_argument("--valid_file", type=str, default="dataset/valid.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<sep>']}
    )
    return tokenizer

def get_model(checkpoint: str, device: str, tokenizer: AutoTokenizer) -> AutoModel:
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, config=config)
    
    
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

if __name__ == "__main__":
    args = parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)
    
    tokenizer = get_tokenizer(args.qg_model)
    
    label2id = {
        "Chi phí quản lý": 1,
        "Chi vật tư, dịch vụ đã sử dụng cho các hoạt động": 2,
        "Chi tiền lương, tiền công và chi phí khác cho người lao động": 3,
        "Chi khác cho các hoạt động": 4,
        "Chi tiền gửi trả lương": 5,
        "Chi bổ sung thu nhập ": 6,
        "Chi phí tài chính": 7,
        "Nộp bảo hiểm": 8,
        "Chi khen thưởng": 9,
        "Nộp kinh phí công đoàn": 10,
        "Nộp thuế thu nhập cá nhân": 11,
        "Nộp thuế thu nhập doanh nghiệp": 12,
        "Chuyển khoản kho bạc chi vật tư, dịch vụ đã sử dụng cho các hoạt động": 13,
        "Chuyển khoản kho bạc chi tiền lương, tiền công và chi phí khác cho người lao động": 14,
        "Chuyển khoản kho bạc chi khác cho các hoạt động": 15,
        "Chuyển khoản kho bạc tiền mua sắm TSCĐ": 16,
        "Chuyển khoản lương vào tài khoản tiền gửi": 17,
        "Chuyển khoản tiền bảo hiểm": 18,
        "Chuyển khoản tiền KPCĐ": 19,
        "Rút dự toán đã tiết kiệm chi về tài khoản tiền gửi ": 20,
        "Rút dự toán về tài khoản tiền gửi": 21,
        "Tạm ứng cho người lao động trong đơn vị hoặc cho đầu mối chi tiêu": 22,
        "Nộp các khoản thuế khác": 23,
        "Nộp phí, lệ phí vào NSNN": 24,
        "Nộp thuế GTGT": 25,
        "Thanh toán tạm ứng cho người lao động trong đơn vị hoặc cho đầu mối chi tiêu": 26,
        "Thu tiền bán sản phẩm, hàng hóa, cung cấp dịch vụ": 27,
        "Thu tiền đã tạm ứng cho người lao động": 28,
        "Thu tiền bán hồ sơ thầu, thanh lý TSCĐ được để lại đơn vị ": 29,
        "Thu tiền khoản thu hộ, chi hộ": 30,
        "Thu doanh thu tài chính": 31,
        "Thu tiền tạm ứng KP KCB từ BHYT": 32,
        "Chuyển khoản kho bạc mua VTHH": 33
    }
 
    train_set = QGDataset(
        json_file=args.train_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer,
        label2id=label2id
    )

    valid_set = QGDataset(
        json_file=args.valid_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer,
        label2id=label2id
    )
    
    for lr in args.learning_rates:
        print(f"Training with learning rate: {lr}")
        model = get_model(args.qg_model, args.device, tokenizer)
        trainer = Trainer(
            dataloader_workers=args.dataloader_workers,
            device=args.device,
            epochs=args.epochs,
            learning_rate=lr,
            model=model,
            pin_memory=args.pin_memory,
            save_dir=args.save_dir,
            tokenizer=tokenizer,
            train_batch_size=args.train_batch_size,
            train_set=train_set,
            valid_batch_size=args.valid_batch_size,
            valid_set=valid_set,
            log_file=args.log_file
        )
        trainer.train()
