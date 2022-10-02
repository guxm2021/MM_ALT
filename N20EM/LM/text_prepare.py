import os


def prepare_text():
    # prepare text corpora for valid and test splits
    valid_path = "data/valid.txt"
    test_path = "data/test.txt"
    
    # valid
    with open(os.path.join("data/n20em/dev.txt"), "r") as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if len(line) > 0]
        print(f"load from data/n20em/dev.txt, read {len(lines)} lines")
    
    with open(valid_path, "w") as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    
    # test
    with open(os.path.join("data/n20em/test.txt"), "r") as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if len(line) > 0]
        print(f"load from data/n20em/test.txt, read {len(lines)} lines")
    
    with open(test_path, "w") as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    
    # prepare text corpora for train splits
    train_path = "data/train.txt"

    # train
    train_lines = []
    for path in ["data/librispeech/train-clean-100.txt", "data/librispeech/train-clean-360.txt", "data/librispeech/train-other-500.txt", "data/dsing/train30.txt", "data/n20em/train.txt"]:
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.split('\n')[0] for line in lines if len(line) > 0]
            train_lines.extend(lines)
            print(f"load from {path}, read {len(lines)} lines")

    with open(train_path, "w") as f:
        for line in train_lines:
            f.write(line)
            f.write('\n')


if __name__ == "__main__":
    prepare_text()
