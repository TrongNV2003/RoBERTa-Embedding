import torch
import numpy as np
import random
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
import csv
from utils import AverageMeter
import torch.nn.functional as F


class Logger:
    def __init__(self, file_path: str, fieldnames: list):
        self.file_path = file_path
        self.fieldnames = fieldnames
        with open(self.file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, data: dict):
        rounded_data = {key: round(value, 3) if isinstance(value, float) else value for key, value in data.items()}
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(rounded_data)

class Trainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        log_file: str,
        # valid_set: Dataset,
        evaluate_on_accuracy: bool = False
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.fieldnames = ['epoch', 'train_loss', 'valid_loss', 'valid_accuracy']

        self.logger = Logger(file_path=log_file, fieldnames=self.fieldnames)

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        # self.valid_loader = DataLoader(
        #     valid_set,
        #     batch_size=valid_batch_size,
        #     num_workers=dataloader_workers,
        #     pin_memory=pin_memory,
        #     shuffle=False
        # )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loss = AverageMeter()
        self.evaluate_on_accuracy = evaluate_on_accuracy
        if evaluate_on_accuracy:
            self.best_valid_score = 0
        else:
            self.best_valid_score = float("inf")

    def train(self) -> None:        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    text_input_ids = data["text_input_ids"].to(self.device)
                    text_attention_mask = data["text_attention_mask"].to(self.device)
                    label_input_ids = data["label_input_ids"].to(self.device)
                    label_attention_mask = data["label_attention_mask"].to(self.device)
                    labels = data["label"].to(self.device)

                    text_embeddings = self.model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
                    label_embeddings = self.model(input_ids=label_input_ids, attention_mask=label_attention_mask).last_hidden_state[:, 0, :]

                    loss = self.cosine_similarity_loss(text_embeddings, label_embeddings, labels)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)
                self._save()

            # if self.evaluate_on_accuracy:
            #     valid_accuracy = self.evaluate_accuracy(self.valid_loader)
            #     if valid_accuracy > self.best_valid_score:
            #         print(
            #             f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_accuracy:.4f}. Saving."
            #         )
            #         self.best_valid_score = valid_accuracy
            #         self._save()
            #     valid_loss = self.evaluate(self.valid_loader)
            #     if valid_loss < self.best_valid_score:
            #         print(
            #             f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
            #         self.best_valid_score = valid_loss
            #         self._save()
            #     self.logger.log({'epoch': epoch, 'train_loss': self.train_loss.avg,
            #                      'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy})
                
            
            # valid_loss = self.evaluate(self.valid_loader)
            # if valid_loss < self.best_valid_score:
            #     print(
            #         f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
            #     self.best_valid_score = valid_loss
            #     self._save()
            # self.logger.log({'epoch': epoch, 'train_loss': self.train_loss.avg,
            #                     'valid_loss': valid_loss, 'valid_accuracy': None})


    def cosine_similarity_loss(self, text_embeddings, label_embeddings, labels, margin=0.5):
        cosine_sim = F.cosine_similarity(text_embeddings, label_embeddings)

        # Tính BCE loss với nhãn 1 cho positive và 0 cho negative
        loss = F.binary_cross_entropy_with_logits(cosine_sim, labels)

        return loss

    # def contrastive_loss(self, text_embeddings, label_embeddings, labels, margin=0.5):
    #     cosine_sim = torch.nn.functional.cosine_similarity(text_embeddings, label_embeddings)
    #     loss = torch.nn.BCEWithLogitsLoss()(cosine_sim, labels.float())
    #     return loss

    # @torch.no_grad()
    # def evaluate(self, dataloader: DataLoader) -> float:
    #     self.model.eval()
    #     eval_loss = AverageMeter()
    #     with tqdm(total=len(dataloader), unit="batches") as tepoch:
    #         tepoch.set_description("validation")
    #         for data in dataloader:
    #             # data = {key: value.to(self.device) for key, value in data.items()}
    #             # output = self.model(**data)
                
    #             text_input_ids = data["text_input_ids"].to(self.device)
    #             text_attention_mask = data["text_attention_mask"].to(self.device)
    #             label_input_ids = data["label_input_ids"].to(self.device)
    #             label_attention_mask = data["label_attention_mask"].to(self.device)
    #             labels = data["label"].to(self.device)

    #             text_embeddings = self.model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
    #             label_embeddings = self.model(input_ids=label_input_ids, attention_mask=label_attention_mask).last_hidden_state[:, 0, :]

    #             loss = self.cosine_similarity_loss(text_embeddings, label_embeddings, labels)
                
    #             eval_loss.update(loss.item(), self.valid_batch_size)
    #             tepoch.set_postfix({"valid_loss": eval_loss.avg})
    #             tepoch.update(1)
    #     return eval_loss.avg

    # @torch.no_grad()
    # def evaluate_accuracy(self, dataloader: DataLoader) -> float:
    #     self.model.eval()
    #     accuracy = AverageMeter()
    #     with tqdm(total=len(dataloader), unit="batches") as tepoch:
    #         tepoch.set_description("validation")
    #         for data in dataloader:
    #             data = {key: value.to(self.device) for key, value in data.items()}
    #             output = self.model(**data)
    #             preds = torch.argmax(output.logits, dim=1)
    #             score = accuracy_score(data["labels"].cpu(), preds.cpu())
    #             accuracy.update(score, self.valid_batch_size)
    #             tepoch.set_postfix({"valid_acc": accuracy.avg})
    #             tepoch.update(1)
    #     return accuracy.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
