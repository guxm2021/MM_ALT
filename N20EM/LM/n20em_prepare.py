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


def prepare_text_by_songs(
    root,
    save_folder,
    skip_prep = False,
):
    """
    This function prepares the text corpora for N20EM single-modality dataset
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

    train_text = []
    dev_text = []
    test_text = []

    for key in data.keys():
        # fetch values
        value = data[key]
        # source_path = value["path"]   # data/<split>/<name>
        split = value["split"]
        wrds = value["lyrics"]

        # append
        if split == "train":
            train_text.append(wrds)
        elif split == "valid":
            dev_text.append(wrds)
        elif split == "test":
            test_text.append(wrds)

        
    # create csv files for each split
    txt_save_train = os.path.join(save_folder, "train.txt")
    with open(txt_save_train, mode="w") as f:
        for line in train_text:
            f.write(line)
            f.write('\n')
    
    txt_save_dev = os.path.join(save_folder, "dev.txt")
    with open(txt_save_dev, mode="w") as f:
        for line in dev_text:
            f.write(line)
            f.write('\n')
    
    txt_save_test = os.path.join(save_folder, "test.txt")
    with open(txt_save_test, mode="w") as f:
        for line in test_text:
            f.write(line)
            f.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/path/to/n20em", help="The saved path for N20EM folder")
    parser.add_argument("--save_folder", type=str, default="data/n20em", help="The saved path for prepared text corpora")
    args = parser.parse_args()

    data_folder = os.path.join(args.data_folder, "utterance_level")
    save_folder = args.save_folder
    # prepare clean data
    prepare_text_by_songs(root=data_folder, save_folder=save_folder)
