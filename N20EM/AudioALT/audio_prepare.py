import re
import os
import csv
import random
import json
import argparse
from collections import Counter
import logging
import torchaudio
from speechbrain.utils.data_utils import download_file, get_all_files
logger = logging.getLogger(__name__)
OPT_FILE = "opt_dsing_prepare.pkl"
SAMPLERATE = 16000


def prepare_alt_by_songs(
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

    csv_lines_train = [["ID", "duration", "wav", "wrd"]]
    csv_lines_valid = [["ID", "duration", "wav", "wrd"]]
    csv_lines_test = [["ID", "duration", "wav", "wrd"]]

    for key in data.keys():
        # fetch values
        value = data[key]
        # source_path = value["path"]   # data/<split>/<name>
        split = value["split"]
        wrds = value["lyrics"]

        # determine target path
        downsample_path = os.path.join(data_folder, key, 'downsample_audio.wav')

        # load audio
        downsample_signal, fs = torchaudio.load(downsample_path)
        assert fs == SAMPLERATE
        duration = downsample_signal.shape[1] / SAMPLERATE
        
        # construct csv files
        csv_line = [
            key, str(duration), downsample_path, wrds,
        ]

        # append
        if split == "train":
            csv_lines_train.append(csv_line)
        elif split == "valid":
            csv_lines_valid.append(csv_line)
        elif split == "test":
            csv_lines_test.append(csv_line)

        
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


def prepare_alt_by_songs_accompany(
    root,
    save_folder = "data/accompany",
    skip_prep = False,
):
    """
    This function prepares the csv files for N20EM single-modality dataset along with accompany
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

    csv_lines_train = [["ID", "duration", "wav", "accomp", "wrd"]]
    csv_lines_valid = [["ID", "duration", "wav", "accomp", "wrd"]]
    csv_lines_test = [["ID", "duration", "wav", "accomp", "wrd"]]
    
    max_value = 0
    min_value = 0
    for key in data.keys():
        # fetch values
        value = data[key]
        # source_path = value["path"]   # data/<split>/<name>
        split = value["split"]
        wrds = value["lyrics"]

        # determine target path
        downsample_path = os.path.join(data_folder, key, 'downsample_audio.wav')

        # load audio
        downsample_signal, fs = torchaudio.load(downsample_path)
        assert fs == SAMPLERATE
        duration = downsample_signal.shape[1] / SAMPLERATE

        resample_accomp_path = os.path.join(data_folder, key, 'downsample_accomp.wav')
        resample_accomp_signal, fs2 = torchaudio.load(resample_accomp_path)
        assert fs2 == SAMPLERATE
        diff =  downsample_signal.shape[1] - resample_accomp_signal.shape[1]
        # diff = abs(diff)
        if diff > max_value:
            print(key)
            print(diff)
            max_value = diff
        if diff < min_value:
            print(key)
            print(diff)
            min_value = diff
        
        # construct csv files
        csv_line = [
            key, str(duration), downsample_path, resample_accomp_path, wrds,
        ]

        # append
        if split == "train":
            csv_lines_train.append(csv_line)
        elif split == "valid":
            csv_lines_valid.append(csv_line)
        elif split == "test":
            csv_lines_test.append(csv_line)

        
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
    prepare_alt_by_songs(root=args.data_folder)
    prepare_alt_by_songs_accompany(root=args.data_folder)
