# Automatic Lyric Transcription with N20EM
This sub-project contains recipes for training audio-only ALT system for the N20EM Dataset. We assume you have downloaded and pre-processed N20EM dataset. The N20EM dataset is saved at `/path/to/n20em`.

## Prerequisites
Please refer to `recipes/N20EM/LM/README.md` to train and save language model before running following experiments. The trained RNNLM is saved at `/path/to/RNNLM`. To train the audio-only ALT system for N20EM dataset (w. DSing), please refer to `recipes/N20EM/ALT/README.md` to train a model on DSing dataset firstly.

## How to run

1. Prepare N20EM dataset, run:
```
python audio_prepare.py --data_folder /path/to/n20em
```

2. Train the audio-only ALT system for N20EM dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_with_wav2vec.py hparams/train_with_wav2vec.yaml --data_parallel_backend --data_folder /path/to/n20em --pretrained_lm_path /path/to/RNNLM --pretrain_dsing False --attempt 1
```

3. Train the audio-only ALT system for N20EM dataset with pretrained model on DSing dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_with_wav2vec.py hparams/train_with_wav2vec.yaml --data_parallel_backend --data_folder /path/to/n20em --pretrained_lm_path /path/to/RNNLM --pretrain_dsing True --attempt 2
```