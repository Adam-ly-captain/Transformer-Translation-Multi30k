import yaml
import os

class ConfigParser:
    
    def __init__(self, config_path="./config/"):
        self.path = config_path
        
    def get_log_config(self):
        # parse log config from yaml file
        return self.get_config(file_name="system.yaml", root_key="log")

    def get_model_config(self, model_name):
        return self.get_config(file_name="model.yaml", root_key=model_name)

    def get_dataset_config(self, dataset_name):
        return self.get_config(file_name="dataset.yaml", root_key=dataset_name)

    def get_config(self, file_name, root_key=None):
        if os.path.isfile(self.path + file_name) is False:
            raise FileNotFoundError(f"Config file {self.path + file_name} not found.")
        
        # parse entire config from yaml file
        with open(self.path + file_name, 'r') as f:
            config = yaml.safe_load(f)

        if root_key:
            return config.get(root_key, {})
        
        return config
