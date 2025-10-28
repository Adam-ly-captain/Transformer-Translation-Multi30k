from torchtext.datasets import Multi30k
import argparse
import torch

from utils.logger import Logger
from utils.parser import ConfigParser
from utils.seed import seed_everything, seed_worker
import models as model_zoo
import dataloaders as dataloader_zoo
import trainer as trainer_zoo


def log(message):
    Logger().log(message)
    

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--mode', type=str, default='test', help='The mode to run.')
    parser.add_argument('--dataset', type=str, default='Multi30k', help='The dataset to use.')
    parser.add_argument('--model', type=str, default='transformer', help='The model to use.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use a pretrained model.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
     
    parser = ConfigParser(config_path=args['config'])
    logger_config = parser.get_log_config()
    model_config = parser.get_model_config(model_name=args['model'])
    dataset_config = parser.get_dataset_config(dataset_name=args['dataset'])
    Logger(filepath=logger_config.get("path", "./logs/"))
    generator = seed_everything(seed=model_config.get("seed", 2025))

    return args, model_config, dataset_config, generator


if __name__ == "__main__":
    args, model_config, dataset_config, generator = init()

    # get dataset iterators
    train_iter = Multi30k(split='train', language_pair=('en', 'de'))
    valid_iter = Multi30k(split='valid', language_pair=('en', 'de'))
    test_iter = Multi30k(split='test', language_pair=('en', 'de'))

    dataset_class = getattr(dataloader_zoo, dataset_config.get("dataset_class_name"))
    loader_class = getattr(dataloader_zoo, dataset_config.get("loader_class_name"))

    train_dataset = dataset_class(data=list(train_iter), dataset_type='train', **dataset_config)
    valid_dataset = dataset_class(data=list(valid_iter), dataset_type='valid', **dataset_config)
    test_dataset = dataset_class(data=list(test_iter), dataset_type='test', **dataset_config)
    
    train_loader = loader_class(
        dataset=train_dataset,
        batch_size=model_config.get("batch_size", 32),
        num_workers=model_config.get("num_workers", 4),
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker
    )
    train_loader.log("Training data loader initialized. Number of records: {}".format(len(train_loader.dataset)))
    valid_loader = loader_class(
        dataset=valid_dataset,
        batch_size=model_config.get("batch_size", 32),
        num_workers=model_config.get("num_workers", 4),
        shuffle=False,
        generator=generator,
        worker_init_fn=seed_worker
    )
    valid_loader.log("Validation data loader initialized. Number of records: {}".format(len(valid_loader.dataset)))
    test_loader = loader_class(
        dataset=test_dataset, 
        batch_size=model_config.get("batch_size", 32),
        num_workers=model_config.get("num_workers", 4),
        shuffle=False,
        generator=generator,
        worker_init_fn=seed_worker
    )
    test_loader.log("Test data loader initialized. Number of records: {}".format(len(test_loader.dataset)))

    # init model instance
    model_class = getattr(model_zoo, model_config.get("model_class_name"))
    model = model_class(args['model'], dataset_config=train_loader.get_dataset_config(), **model_config)
    if args['pretrained']:
        model.load(model_config.get("pretrained_path", ""))

    trainer_class = getattr(trainer_zoo, model_config.get("trainer_class_name"))
    trainer = trainer_class(
        model=model,
        train_loader=train_loader, 
        val_loader=valid_loader, 
        test_loader=test_loader,
        **model_config
    )
    log(f"Model {args['model']} initialized and parameters count: {model.count_parameters()}")

    # start training/testing
    if args['mode'] == 'train':
        trainer.fit()
    elif args['mode'] == 'test':
        trainer.test()
    else:
        raise ValueError(f"Unsupported mode: {args['mode']}")
