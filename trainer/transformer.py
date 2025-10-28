from .base import BaseTrainer
from utils.metrics import NLPMetrics

import tqdm
import torch


class TransformerTrainer(BaseTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def fit(self):
        self.model.train()
        total_loss = 0

        self.log(f"Starting training ...", mode='train')
        min_val_loss = float('inf')
        max_val_bleu = float('-inf')
        epochs_no_improve = 0
        for epoch in range(self.model.current_epoch, self.num_epochs):
            epoch_loss = 0
            for batch in tqdm.tqdm(self.train_loader, desc="Training"):
                src, tgt_shift, tgt = batch
                src, tgt_shift, tgt = src.to(self.device), tgt_shift.to(self.device), tgt.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model((src, tgt_shift, tgt), mode='train')
                loss = self.criterion(logits, tgt, ignore_index=self.train_loader.get_dataset_config()['pad_idx'])  # softmax + cross-entropy with masking
                loss.backward()
                # gradient clipping
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    
                self.optimizer.step()

                epoch_loss += loss.item()

            self.log(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {epoch_loss / len(self.train_loader)}", mode='train')
            total_loss += epoch_loss
            self.model.current_epoch = epoch + 1
            
            if (epoch + 1) % self.eval_interval == 0:
                val_loss, bleu_scores = self.evaluate()
                self.log(f"Epoch {epoch + 1}/{self.num_epochs} - Validation Loss: {val_loss} - BLEU Score: {bleu_scores}", mode='eval')

                if val_loss < min_val_loss or bleu_scores > max_val_bleu:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                    if bleu_scores > max_val_bleu:
                        max_val_bleu = bleu_scores
                        
                    epochs_no_improve = 0
                    self.model.save()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.early_stopping_patience:
                    self.log(f"Early stopping triggered. No improvement for {self.early_stopping_patience} epochs.", mode='train')
                    break

        return total_loss / self.num_epochs
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        bleu_scores = 0
        with torch.no_grad():
            self.log(f"Starting evaluation ...", mode='eval')
            for batch in tqdm.tqdm(self.val_loader, desc="Evaluating"):
                src, tgt_shift, tgt = batch
                src, tgt_shift, tgt = src.to(self.device), tgt_shift.to(self.device), tgt.to(self.device)

                generated, logits = self.model((src, tgt_shift, tgt), mode='predict')
                loss = self.criterion(logits, tgt, ignore_index=self.val_loader.get_dataset_config()['pad_idx'])
                total_loss += loss.item()
                
                bleu_scores += NLPMetrics.bleu_score_batch(
                    references=tgt, hypotheses=generated, 
                    sos_token_id=self.val_loader.get_dataset_config()['sos_idx'], 
                    eos_token_id=self.val_loader.get_dataset_config()['eos_idx'], 
                    pad_token_id=self.val_loader.get_dataset_config()['pad_idx']
                )

        return total_loss / len(self.val_loader), bleu_scores / len(self.val_loader)
        
    def test(self):
        self.model.eval()
        bleu_scores = 0
        with torch.no_grad():
            self.log(f"Starting testing ...", mode='test')
            for batch in tqdm.tqdm(self.test_loader, desc="Testing"):
                src, tgt_shift, tgt = batch
                src, tgt_shift, tgt = src.to(self.device), tgt_shift.to(self.device), tgt.to(self.device)

                generated, logits = self.model((src, tgt_shift, tgt), mode='predict')
                
                bleu_scores += NLPMetrics.bleu_score_batch(
                    references=tgt, hypotheses=generated,
                    sos_token_id=self.test_loader.get_dataset_config()['sos_idx'],
                    eos_token_id=self.test_loader.get_dataset_config()['eos_idx'],
                    pad_token_id=self.test_loader.get_dataset_config()['pad_idx']
                )

        bleu_scores /= len(self.test_loader)
        self.log(f"Test BLEU-4 Score: {bleu_scores}", mode='test')
            
        return bleu_scores
