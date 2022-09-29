# Language Model with DSing
This sub-project contains recipes for training RNN langugae model for the DSing Dataset. We assume you have downloaded and pre-processed DSing dataset. The DSing dataset is saved at `/path/to/DSing`.

## How to run

1. Prepare Text corpora, run:
```
python text_prepare.py --data_folder /path/to/DSing --duration_threshold 28
```
To facilitate the training of ALT, we eliminate the utterances whose durations are longer than 28s and the utterances whose transcriptions contain numbers in the training set. You may want to change the `duration_threshold` based on your GPUs.

2. Train the RNNLM for DSing dataset, run:
```
python train_rnnlm.py hparams/train_rnnlm.yaml --duration_threshold 28
```
The results are saved to `results/RNNLM_duration28/<seed>/CKPT-files`. We mark the CKPT folder of best model is `/path/to/RNNLM`. More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).


## Results
| Release | hyperparams file | Tokenizer | Val. loss | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 22-09-29 | train_rnnlm.yaml |  Character | 0.809 | https://drive.google.com/drive/folders/1VDyVV-ksX2hLEAv8pWE-LjmUTbDhU2zu?usp=sharing | 1xA5000 23GB |