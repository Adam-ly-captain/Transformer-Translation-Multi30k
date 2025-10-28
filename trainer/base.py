from utils.logger import Logger
from utils.criterion import get_criterion
from utils.optimizer import get_optimizer

import tqdm
import torch


class BaseTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        test_loader,  
        **kwargs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = kwargs.get("device", "cpu")
        self.model.to(self.device)  # load weights to device
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.optimizer = get_optimizer(kwargs.get("optimizer", 'adam'), model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = get_criterion(kwargs.get("criterion", 'mse'))
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.eval_interval = kwargs.get("eval_interval", 1)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 5)
        self.max_grad_norm = kwargs.get("max_grad_norm", None)

    def fit(self):
        self.model.train()
        total_loss = 0

        self.log(f"Starting training ...", mode='train')
        min_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.model.current_epoch, self.num_epochs):
            epoch_loss = 0
            for batch in tqdm.tqdm(self.train_loader, desc="Training"):
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model((data, labels))
                loss = self.criterion(y_pred, labels)
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
                val_loss = self.evaluate()
                self.log(f"Epoch {epoch + 1}/{self.num_epochs} - Validation Loss: {val_loss}", mode='eval')

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
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

        with torch.no_grad():
            self.log(f"Starting evaluation ...", mode='eval')
            for batch in tqdm.tqdm(self.val_loader, desc="Evaluating"):
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                output = self.model((data, labels))
                loss = self.criterion(output, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def test(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            self.log(f"Starting testing ...", mode='test')
            for batch in tqdm.tqdm(self.test_loader, desc="Testing"):
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                output = self.model((data, labels))
                loss = self.criterion(output, labels)

                total_loss += loss.item()

        return total_loss / len(self.test_loader)
    
    def log(self, message, mode='train', save_to_file=False):
        Logger().log(f"Model {self.model.model_name} [{mode}] : {message}")
        if save_to_file:
            Logger().save(message)
            
    