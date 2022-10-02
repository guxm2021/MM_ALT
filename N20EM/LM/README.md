# Language Model with N20EM
This sub-project contains recipes for training RNN langugae model for the N20EM Dataset. We assume you have downloaded and pre-processed N20EM dataset. The N20EM dataset is saved at `/path/to/n20em`.

## Prerequisites
We train RNNLM on LibriSpeech&DSing&N20EM training splits and evaluate the model on N20EM valid&test splits. Please follow the step 1-4 to prepare the data.

1. Prepare text corpus for LibriSpeech, run:
```
python librispeech_prepare.py --data_folder /path/to/librispeech
```
add `--download` if you need to download librispeech dataset, ~1,000 h speech dataset in total.

2. Prepare text corpus for DSing, run:
```
python dsing_prepare.py --data_folder /path/to/dsing
```

3. Prepare text corpus for N20EM, run:
```
python n20em_prepare.py --data_folder /path/to/n20em
```

4. Combine text corpus from three datasets into train split, run:
```
python text_prepare.py
```


## How to run

Train the RNNLM for N20EM dataset, run:
```
python -m torch.distributed.launch --nproc_per_node=4 train_rnnlm.py hparams/train_rnnlm.yaml --distributed_launch --distributed_backend='nccl'
```
The results are saved to `results/train_rnnlm_attempt1/<seed>/CKPT-files`. We mark the CKPT folder of best model is `/path/to/RNNLM`. More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).

## Results
| Release | hyperparams file | Tokenizer | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| --------:| :-----------:|
| 22-10-02 | train_rnnlm.yaml |  Character | https://drive.google.com/drive/folders/1XAzFetSLAZ77EdsiM_N28wV62q64odsJ?usp=sharing | 4xA5000 23GB |