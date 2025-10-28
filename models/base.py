import os
import torch.nn as nn
import torch
from datetime import datetime

from utils.logger import Logger
from utils.parser import ConfigParser

class BaseComponent(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")


class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.config_parser = ConfigParser()
        self.model_name = model_name
        self.current_epoch = 0
    
    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def save(self, checkpoint_file_name=""):
        checkpoint_path = self.config_parser \
                            .get_model_config(model_name=self.model_name) \
                            .get("checkpoint_path", "./results/checkpoints/")
                            
        if os.path.exists(checkpoint_path) is False:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        if not checkpoint_file_name:
            now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            checkpoint_file_name = f"{self.model_name}_{str(self.current_epoch)}_{now}_checkpoint.pt"
        
        if not checkpoint_file_name.endswith(".pt"):
            checkpoint_file_name += ".pt"
        
        if not checkpoint_path.endswith("/"):
            checkpoint_path += "/"

        full_path = os.path.join(checkpoint_path, checkpoint_file_name)
        
        # Save the model checkpoint
        torch.save(self.state_dict(), full_path)
        self.log(f"Model saved to {full_path}")

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)

    def load(self, checkpoint_file_path):
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_file_path} does not exist.")
        
        self.load_state_dict(torch.load(checkpoint_file_path))
        self.current_epoch = int(checkpoint_file_path.split("_")[-3])
        self.log(f"Model loaded from {checkpoint_file_path}")

    def log(self, message, save_to_file=False):
        Logger().log(f"{self.model_name} invoking : {message}")
        if save_to_file:
            Logger().save(message)
            
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
