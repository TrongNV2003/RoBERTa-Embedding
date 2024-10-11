import argparse
from transformers import AutoConfig, RobertaModel, AutoTokenizer
from data_loader import QGDataset
from trainer import Trainer
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--qa_model", type=str, default="./bkai-encoder-labeling")
    parser.add_argument("--qa_mcq_model", type=str, default="./t5-base-question-mcq-generator")
    parser.add_argument("--distractor_model", type=str, default="./t5-base-distractor-generator")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="testing.csv")
    return parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    return tokenizer

def get_model(checkpoint: str, device: str, tokenizer: AutoTokenizer) -> RobertaModel:
    config = AutoConfig.from_pretrained(checkpoint)
    model = RobertaModel.from_pretrained(checkpoint, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

def main():
    args = parse_args()

    model_configs = [
        {"type": "Model sentences", "checkpoint": args.qa_model, "dataset_class": QGDataset},
    ]

    results = []

    for config in model_configs:
        tokenizer = get_tokenizer(config["checkpoint"])
        train_set = config["dataset_class"](
            json_file='dataset/train.json',
            max_length=args.max_length,
            pad_mask_id=args.pad_mask_id,
            tokenizer=tokenizer
        )
        test_set = config["dataset_class"](
            json_file='dataset/test.json',
            pad_mask_id=args.pad_mask_id,
            max_length=args.max_length,
            tokenizer=tokenizer
        )
        model = get_model(config["checkpoint"], args.device, tokenizer)
        trainer = Trainer(
            dataloader_workers=args.dataloader_workers,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model=model,
            tokenizer=tokenizer,
            pin_memory=args.pin_memory,
            save_dir="",
            train_batch_size=args.train_batch_size,
            train_set=train_set,
            valid_batch_size=args.test_batch_size,
            valid_set=test_set,
            log_file=args.log_file,
            evaluate_on_accuracy=True
        )

        loss = trainer.evaluate(trainer.valid_loader)
        accuracy = trainer.qg_accuracy(trainer.valid_loader)
        results.append({"model_type": config["type"], "loss": round(loss, 3), "accuracy": round(accuracy*10, 3)})

    for result in results:
        print(f"Model type: {result['model_type']}")
        print(f"Loss: {result['loss']}")
        print(f"Accuracy: {result['accuracy']}")

    # Lưu kết quả vào file CSV
    df = pd.DataFrame(results)
    df.to_csv(args.log_file, index=False)
    print(f"Results saved to {args.log_file}")
    
if __name__ == "__main__":
    main()
