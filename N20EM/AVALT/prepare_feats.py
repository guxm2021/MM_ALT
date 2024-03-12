import os
import csv
import json
import torch
import logging
import argparse
import torchaudio
logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_features(
    feat1_path,
    feat2_path,
    json_path,
    save_folder = "data/feats",
    skip_prep = False,
):
    if skip_prep:
        return
    
    # save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # open json files
    with open(json_path, 'r') as f:
        data = json.load(f)
    f.close()

    csv_lines_train = [["ID", "frame1", "frame2", "audio", "video", "wrd"]]
    csv_lines_valid = [["ID", "frame1", "frame2", "audio", "video", "wrd"]]
    csv_lines_test = [["ID", "frame1", "frame2", "audio", "video", "wrd"]]

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
        audio_path = os.path.join(feat1_path, "clean", key + ".pt")
        video_path = os.path.join(feat2_path, key + ".pt")
        
        # load
        audio_feat = torch.load(audio_path)
        video_feat = torch.load(video_path)

        frame1 = audio_feat.shape[0]
        frame2 = video_feat.shape[0]

        # test 
        diff = frame1 - frame2 * 2
        if diff > max_diff:
            print(diff)
            max_diff = diff
        if diff < min_diff:
            print(diff)
            min_diff = diff
        
        # construct csv files
        csv_line = [
            key, str(frame1), str(frame2), audio_path, video_path, wrds,
        ]

        # append
        # append
        if split == "train":
            csv_lines_train.append(csv_line)
        elif split == "valid":
            csv_lines_valid.append(csv_line)
        elif split == "test":
            csv_lines_test.append(csv_line)
    
    print(max_diff)
    print(min_diff) 
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
    
    feat1_path = "../AudioALT/data/feats"
    feat2_path = "../VideoALT/data/feats"
    json_path = os.path.join(args.data_folder, "metadata_split_by_song.json")
    prepare_features(feat1_path, feat2_path, json_path)