# Transformer-Translation-Multi30k

## Project Overview
```bash
./
├── config
│   ├── dataset.yaml
│   ├── model.yaml
│   └── system.yaml
├── data
│   └── multi30k
│       └── processed
│           ├── src_vocab.txt
│           └── tgt_vocab.txt
├── dataloaders
│   ├── __init__.py
│   ├── base.py
│   └── mmt.py
├── logs
├── main.py
├── models
│   ├── __init__.py
│   ├── base.py
│   └── transformer.py
├── results
│   └── checkpoints
├── test.py
├── trainer
│   ├── __init__.py
│   ├── base.py
│   └── transformer.py
└── utils
```

Above is the project structure for a Transformer-based machine translation model using the Multi30k dataset. I provide the preprocessed data and vocabulary files in the `data/multi30k/processed` directory. The raw dataset can be downloaded from the [Multi30k Dataset](https://github.com/multi30k/dataset) or loaded from the torchtext library.

## Environment Setup

* python >= 3.10

```bash
pip install -r requirements.txt
```

## Usage
To train the model, run:

```bash
python main.py --mode train --gpu 0
```

To test the model, run:

```bash
python main.py --mode test --gpu 0
```