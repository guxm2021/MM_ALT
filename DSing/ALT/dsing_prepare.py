import re
import os
import csv
import random
from collections import Counter
import logging
import argparse
import torchaudio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
)
logger = logging.getLogger(__name__)
OPT_FILE = "opt_dsing_prepare.pkl"
SAMPLERATE = 16000


def prepare_dsing(
    data_folder,
    save_folder,
    train_splits = [],
    dev_splits = [],
    test_splits = [],
    dur_threshold = 35,
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    skip_prep=False,
):
    """
    This class prepares the csv files for the DSing dataset

    Arguments
    -------
    data_folder : str 
        Path to the folder where the original DSing dataset is stored.
    save_folder : str
        The directory where to store the csv files
    train_splits : list
        List of train splits to prepare from ['train1','train3','train30'].
    dev_splits : list
        List of dev splits to prepare from ['dev'].
    test_splits : list
        List of test splits to prepare from ['test'].
    dur_threshold: int
        Default: 35
        Duration threshold for filtering wav files
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of DSing to merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.
    """
    if skip_prep:
        return
    data_folder = data_folder
    splits = train_splits + dev_splits + test_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")
    
    # Additional checks to make sure the data folder contains DSing
    check_dsing_folders(data_folder, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):
        split = splits[split_index]
        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".wav"]
        )
        
        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["transcription.txt"]
        )

        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(
            save_folder, wav_lst, text_dict, split, n_sentences, dur_threshold,
        )
    
    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".csv" for split_libri in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )
        
    # saving options
    save_pkl(conf, save_opt)


def create_csv(
    save_folder, wav_lst, text_dict, split, select_n_sentences, dur_threshold,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.
    dur_threshold : int
        Duration Threshold for filtering wav files

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    snt_cnt = 0
    # max_durations = []
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        spk_id = "-".join(snt_id.split("-")[0:2])
        wrds = text_dict[snt_id]

        signal, fs = torchaudio.load(wav_file)
        assert fs == SAMPLERATE     # check the sampling rate
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE
        # if len(max_durations) < 20:
        #     max_durations.append(duration)
        # else:
        #     for index in range(len(max_durations)):
        #         if duration > max_durations[index]:
        #             max_durations.pop(index)
        #             max_durations.append(duration)
        #             break

        # delete the wav files whose durations are too long
        if duration > dur_threshold:
            continue
        
        # delete the wav files whose annotations include numbers
        wrds_seg = wrds.strip().split("_")
        has_number = False
        for word in wrds_seg:
            if bool(re.search(r'\d', word)):
                has_number = True
                break
        if has_number:
            continue

        csv_line = [
            snt_id,
            str(duration),
            wav_file,
            spk_id,
            str(" ".join(wrds.split("_"))),
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)
    #print(max_durations)
    # Final print
    msg = "%s successfully created! using duration threshold %d s" % (csv_file, dur_threshold)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the dsing data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the dsing text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    digits = []
    # Reading all the transcription files is text_lst
    for file in text_lst:
        with open(file, "r") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                # print(line_lst)
                # check the numbers
                annotations = line_lst[1:]
                for anno in annotations:
                    if bool(re.search(r'\d', anno)):
                        digits.append(anno)
                        break
                text_dict[line_lst[0]] = "_".join(annotations)
    print(digits)
    return text_dict


def check_dsing_folders(data_folder, splits):
    """
    Check if the data folder actually contains the DSing dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If DSing is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "DSing dataset)" % split_folder
            )
            raise OSError(err_msg)


if __name__ == "__main__":
    # define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/path/to/DSing", help="The saved path for DSing folder")
    parser.add_argument("--save_folder", type=str, default="data/28", help="The saved path for prepared text corpora")
    parser.add_argument("--duration_threshold", type=int, default=28, help="Duration threshold for filtering wav files")
    args = parser.parse_args()
    data_folder = args.data_folder
    save_folder = args.save_folder
    prepare_dsing(
    data_folder,
    save_folder,
    train_splits = ['train1', 'train3', 'train30'],
    dev_splits = ['dev'],
    test_splits = ['test'],
    dur_threshold=args.dur_threshold,
    select_n_sentences=None,
    skip_prep=False,
)
    