# Lryic Lip Reading with N20EM
This sub-project contains recipes for training video-only ALT system for the N20EM Dataset. We assume you have downloaded and pre-processed N20EM dataset. The N20EM dataset is saved at `/path/to/n20em`.

## Prerequisites
Please refer to `N20EM/LM/README.md` to train and save language model before running following experiments. The trained RNNLM is saved at `/path/to/RNNLM`.

## How to run

1. Prepare N20EM dataset, run:
```
python video_prepare.py --data_folder /path/to/n20em
```

2. Train the video-only ALT system for N20EM dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_avhubert.py hparams/train_avhubert.yaml --data_parallel_backend --data_folder /path/to/n20em --pretrained_lm_path /path/to/RNNLM
```