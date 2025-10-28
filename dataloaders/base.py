import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from utils.logger import Logger


class BaseDataset(Dataset):
    def __init__(self, dataset_type='train', **kwargs):
        super().__init__()
        
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_dataset_type(self):
        return self.dataset_type


class BaseCollator(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, batch):
        return default_collate(batch)


class BaseDataLoader(DataLoader):
    """
    A base data loader class that can be extended for specific datasets.
    """
    def __init__(self, dataset, batch_size=64, shuffle=False,
                 num_workers=1, collate_fn=None, **kwargs):
        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=collate_fn or default_collate, **kwargs)
        
        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
    
    def log(self, message, save_to_file=False):
        Logger().log(f"[{self.dataset.get_dataset_type()} dataset] : {message}")
        if save_to_file:
            Logger().save(message)
            