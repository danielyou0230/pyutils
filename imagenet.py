import argparse
import subprocess
import os
from tqdm import tqdm
"""
Unpacking ImageNet dataset

Extract tarballs under directory ImageNet/
python imagenet.py --dir=ImageNet/ --type=all

ImageNet/ already in under directory with root permission
python imagenet.py --dir=/usr/local/datasets/ImageNet --type=all --sudo

Extract local files to root-owned directory
python imagenet.py --dir=ImageNet/ --destination=/usr/local/datasets/ImageNet --type=all --sudo
"""

SPLITS = ["train", "train_t3", "val", "test"]


def main(args):
    print("ImageNet untarring tool")
    # Get all tarball files
    files = getFile(args.dir, contain=args.year, extension=".tar")
    print("Target: {}".format(SPLITS if args.type == "all" else args.type))

    # Filter out unwanted files
    if args.type in SPLITS:
        postfix = "_{}.tar".format(args.type)
        files = [itr for itr in files if itr.endswith(postfix)]

    # Untar first level (i.e. train/train_t3/test/val)
    batch_untar(files, args.destination, args.sudo, verbose=True)

    if args.type in ["all", "train", "train_t3"]:
        print(" * Unpacking train/train_t3 data")
        # Get training set name
        files = [itr for itr in files if "train" in itr]
        for itr in files:
            # print(files)
            directory = os.path.splitext(itr)[0]
            # Get tarballs under training set
            files = getFile(directory, extension=".tar")
            # Untar train / train_t3
            batch_untar(
                files,
                args.destination,
                args.sudo,
                verbose=args.verbose,
                flatten=args.flatten)


def batch_untar(files, dest=None, sudo=False, verbose=False, flatten=False):
    # Using sudo in scripts is dangerous, use it only when necessary
    prefix = "sudo " if sudo else ""

    # iterable objects: using tqdm in verbose mode otherwise use normal list
    desc = "Progress"
    _files = tqdm(files, desc=desc) if not verbose else files

    # Untar the files
    # pv ILSVRC2012_img_test.tar | sudo tar xf - -C ILSVRC2012_img_test
    for file in _files:
        cmds = list()
        # Check directory exist or overwrite
        if dest is not None:
            name = "" if flatten else os.path.basename(file)
            directory = os.path.join(dest, name)
        elif flatten:
            directory = os.path.dirname(file)
        else:
            directory = os.path.splitext(file)[0]

        # create path for the files in .tar
        if not os.path.isdir(directory):
            cmds.append("{}mkdir {}".format(prefix, directory))
        else:
            pass

        # Untar files
        if verbose:
            print("Untarring: {}".format(file))
            cmds.append("pv {} | {}tar -xf - -C {}".format(
                file, prefix, directory))
        else:
            cmds.append("{}tar -C {} -xf {}".format(prefix, directory, file))

        for cmd in cmds:
            subprocess.run(cmd, shell=True)


def getFile(directory, contain=None, extension=".tar"):
    files = os.listdir(directory)
    files = filter(lambda x: os.path.splitext(x)[1] == extension, files)
    if contain is not None:
        files = filter(lambda x: str(contain) in x, files)
    files = [os.path.join(directory, itr) for itr in files]
    assert len(files) > 0
    return list(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default="./", type=str, help="Directory to the tarballs.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosely untarring files under trainingset.")
    parser.add_argument(
        "--type", choices=["all", "train", "train_t3", "val", "test"])
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten second level of the files in training set.")
    parser.add_argument(
        "--sudo", action="store_true", help="Use sudo to untar")
    parser.add_argument(
        "--destination",
        type=str,
        help="Destination of the files to be extracted to.")
    parser.add_argument(
        "--year",
        type=int,
        help="Year of data to extract.")
    args = parser.parse_args()

    main(args)
