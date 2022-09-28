# Automatic Lyric Transcription with DSing
This sub-project contains recipes for training audio-only ALT system for the DSing Dataset. We assume you have downloaded and pre-processed DSing dataset. The DSing dataset is saved at `/path/to/DSing`.

## Prerequisites
Please refer to `DSing/LM/README.md` to train and save language model before running following experiments. The trained RNNLM is saved at `/path/to/RNNLM`.

## How to run

1. Prepare DSing dataset, run:
```
python dsing_prepare.py --data_folder /path/to/DSing --duration_threshold 28
```

2. Train the audio-only ALT system for DSing dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py hparams/train_wav2vec.yaml --data_parallel_backend --data_folder /path/to/dsing --pretrained_lm_path /path/to/RNNLM --save_model --duration_threshold 28
```
The option `--save_model` is used to separately save the model to the folder `DSing/save_model/` besides the CKPT files for the usage of N20EM experiments.


We use four A5000 GPUs (each has 23 GB) to run experiments. To facilitate the training, we eliminate the utterances longer than 28s in the training set. You may want to change the `duration_threshold` based on your GPUs.


## Results
| Release | hyperparams file | Val. WER | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 22-09-29 | train_with_wav2vec.yaml |  13.26 | 14.56 | - | 4xA5000 23GB |