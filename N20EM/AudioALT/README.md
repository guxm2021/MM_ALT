# Automatic Lyric Transcription with N20EM
This sub-project contains recipes for training audio-only ALT system for the N20EM Dataset. We assume you have downloaded and pre-processed N20EM dataset. The N20EM dataset is saved at `/path/to/n20em`.

## Prerequisites
1. Before runnning our scripts, you need to download, preprocess and save the N20EM properly. For your convenience, we already crop the video clips of lip movements without releasing the identity of each subject.

The file organization for N20EM should be:
```
/path/to/N20EM
├── data
    ├── id1
        ├── downsample_audio.wav
        ├── downsample_accomp.wav
        ├── video.mp4
        ├── imu.csv
    ├── id2
    ├── ...
├── metadata_split_by_song.json
├── README.txt
```

2. Please refer to `recipes/N20EM/LM/README.md` to train and save language model before running following experiments. The trained RNNLM is saved at `/path/to/RNNLM`. To train the audio-only ALT system for N20EM dataset (w. DSing), please refer to `recipes/N20EM/ALT/README.md` to train a model on DSing dataset firstly.

## How to run

1. Prepare N20EM dataset, run:
```
python audio_prepare.py --data_folder /path/to/n20em
```

2. Train the audio-only ALT system for N20EM dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec2.py hparams/train_wav2vec2.yaml --data_parallel_backend --data_folder /path/to/n20em --pretrained_lm_path /path/to/RNNLM --pretrain_dsing False --attempt 1
```

3. Train the audio-only ALT system for N20EM dataset with pretrained model on DSing dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec2.py hparams/train_wav2vec2.yaml --data_parallel_backend --data_folder /path/to/n20em --pretrained_lm_path /path/to/RNNLM --pretrain_dsing True --attempt 2
```
