import re
import os
import cv2
import csv
import random
import json
import argparse
from collections import Counter
import logging
import torchaudio
logger = logging.getLogger(__name__)
OPT_FILE = "opt_dsing_prepare.pkl"
SAMPLERATE = 16000


def prepare_vlt_by_songs(
    root,
    save_folder = "data/songs",
    skip_prep = False,
):
    """
    This function prepares the csv files for N20EM single-modality dataset
    Split the dataset by songs
    """
    if skip_prep:
        return
    data_folder = os.path.join(root, 'data')
    anno_path = os.path.join(root, 'metadata_split_by_song.json')
    
    # save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # open json files
    with open(anno_path, 'r') as f:
        data = json.load(f)
    f.close()

    csv_lines_train = [["ID", "frame", "mp4", "wrd"]]
    csv_lines_valid = [["ID", "frame", "mp4", "wrd"]]
    csv_lines_test = [["ID", "frame", "mp4", "wrd"]]
    # Note: duration for audio / frame for video
    max_diff = 0.0
    min_diff = 0.0
    min_key = None
    max_key = None
    for key in data.keys():
        # fetch values
        value = data[key]
        # source_path = value["path"]   # data/<split>/<name>
        split = value["split"]
        wrds = value["lyrics"]

        # determine target path
        audio_path = os.path.join(data_folder, key, 'downsample_audio.wav')
        video_path = os.path.join(data_folder, key, 'video.mp4')

        # load audio
        audio, fs = torchaudio.load(audio_path)
        assert fs == SAMPLERATE
        duration = audio.shape[1] / SAMPLERATE

        # load video
        frame = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        if frame == 0:
            print(key)
        
        # test
        duration2 = frame / 25
        diff = duration - duration2
        if diff > max_diff:
            print(diff)
            max_diff = diff
            max_key = key
        if diff < min_diff:
            print(diff)
            min_diff = diff
            min_key = key

        # construct csv files
        csv_line = [
            key, str(frame), video_path,  wrds,
        ]

        # append
        if split == "train":
            csv_lines_train.append(csv_line)
        elif split == "valid":
            csv_lines_valid.append(csv_line)
        elif split == "test":
            csv_lines_test.append(csv_line)

    print(max_diff)
    print(max_key) 
    print(min_diff)
    print(min_key)   
    # create csv files for each split
    csv_save_train = os.path.join(save_folder, "n20em_train.csv")
    with open(csv_save_train, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_train:
            csv_writer.writerow(line)
    
    csv_save_valid = os.path.join(save_folder, "n20em_valid.csv")
    with open(csv_save_valid, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_valid:
            csv_writer.writerow(line)
    
    csv_save_test = os.path.join(save_folder, "n20em_test.csv")
    with open(csv_save_test, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_test:
            csv_writer.writerow(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/path/to/N20EM", help="The saved path for N20EM folder")
    args = parser.parse_args()

    # prepare clean data
    prepare_vlt_by_songs(root=args.data_folder)
