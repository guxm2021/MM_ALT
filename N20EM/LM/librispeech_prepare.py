import os
import csv
import random
import shutil
from collections import Counter
import logging
import torchaudio
import argparse
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLERATE = 16000


def prepare_text_corpora(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> tr_splits = ['train-clean-100']
    >>> dev_splits = ['dev-clean']
    >>> te_splits = ['test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
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

    # Additional checks to make sure the data folder contains Librispeech
    check_librispeech_folders(data_folder, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):

        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        )

        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["trans.txt"]
        )

        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_txt(
            save_folder, wav_lst, text_dict, split, n_sentences,
        )

    # saving options
    save_pkl(conf, save_opt)


def create_txt(
    save_folder, wav_lst, text_dict, split, select_n_sentences,
):
    """
    Create the dataset txt file given a list of wav files.

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

    Returns
    -------
    None
    """
    # Setting path for the text file
    text_file = os.path.join(save_folder, split + ".txt")

    # Preliminary prints
    msg = "Creating text lists in  %s..." % (text_file)
    logger.info(msg)

    text_lines = []

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        snt_id = wav_file.split("/")[-1].replace(".flac", "")
        wrds = text_dict[snt_id]
        text_line = str(" ".join(wrds.split("_")))

        #  Appending current file to the csv_lines list
        text_lines.append(text_line)
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    # Writing the txt_lines
    with open(text_file, mode="w") as f:
        for line in text_lines:
            f.write(line)
            f.write('\n')
    #print(max_durations)
    # Final print
    msg = "%s successfully created!" % text_file
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

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
        Path to the file containing the librispeech text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    for file in text_lst:
        with open(file, "r") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def check_librispeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the LibriSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If LibriSpeech is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Librispeech dataset)" % split_folder
            )
            raise OSError(err_msg)


def download_librispeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    archives = ["train-other-500.tar.gz", "train-clean-100.tar.gz", "train-clean-360.tar.gz", "dev-clean.tar.gz", "dev-other.tar.gz", "test-clean.tar.gz", "test-other.tar.gz"]
            
    for archive in archives:
        url =  "https://www.openslr.org/resources/12/" + archive
        print(f"download data from {url}")
        save_path = os.path.join(destination, archive)
        download_file(url, save_path)
        shutil.unpack_archive(save_path, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action='store_true', help="Whether to download Librispeech dataset")
    parser.add_argument("--data_folder", type=str, default="/path/to/librispeech", help="The saved path for Librispeech folder")
    parser.add_argument("--save_folder", type=str, default="data/librispeech", help="The saved path for prepared text corpora")
    args = parser.parse_args()

    data_folder = args.data_folder
    save_folder = args.save_folder
    if args.download:
        download_librispeech(data_folder)
    # prepare clean data
    prepare_text_corpora(data_folder=data_folder, save_folder=save_folder,
                         tr_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                         dev_splits=["dev-clean", "dev-other"], 
                         te_splits=["test-clean", "test-other"])
