# Language Model with DSing
This sub-project contains recipes for training RNN langugae model for the DSing Dataset. We assume you have downloaded and pre-processed DSing dataset. The DSing dataset is saved at `/path/to/DSing`.

## How to run

1. Prepare Text corpora, run:
```
python text_prepare.py --data_folder /path/to/DSing --duration_threshold 28
```

2. Train the RNNLM for DSing dataset, run:
```
python -m torch.distributed.launch --nproc_per_node=4 train_rnnlm.py hparams/train_rnnlm.yaml --distributed_launch --distributed_backend='nccl' --duration_threshold 28
```
The model will be saved as CKPT files, we mark the path to best model as `/path/to/RNNLM`. Here we provide a [trained RNNLM](). If you skip training the RNNLM from scratch, please place it in the right place `/path/to/RNNLM`.  More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).

We use four A5000 GPUs (each has 23 GB) to run experiments. To facilitate the training, we eliminate the utterances longer than 28s in the training set. You may want to change the `duration_threshold` based on your GPUs.