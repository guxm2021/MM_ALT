# Voice Activity Detection from IMU with N20EM
This sub-project contains recipes for implementing voice activity detection from IMU for the N20EM Dataset. We assume you have downloaded and pre-processed N20EM dataset. The N20EM dataset is saved at `/path/to/n20em`.

## How to run

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

2. To train the CRNN model, run the following command:
```
python main.py /path/to/IMU_VAD_data
```

`/path/to/IMU_VAD_data` is the path to save data for the task of Voice Activity Detection from IMU.