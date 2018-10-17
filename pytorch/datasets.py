import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    """
    Generic Pytorch Image dataset object

    Args:
        directory(str): The root path of where the images are stored
            in a hierarchical manner, i.e. the directory name of second
            level directories are the labels of the images store under
            them.
        ext(list or str): The valid target file extensions.
            (default: [".png", "jpeg", ".jpg"])
        dump(bool): Dump the file_list if asserted.
        from_file(bool): Load data from file. Format: [PATH_TO_FILE] \t [LABEL]
        from_list(list): If given, the module assumes the list has the format
            [PATH_TO_FILE] \t [LABEL] in each entry.
        working_dir(str): Working directory for the data, to support data stored
            in different place with the file list.
        output(str): The output filename to dump the file_list under directory
        transforms(torchvision.transform): Customized transformation option.
            [TO-DOs] Support multiple methods.
        verbose(bool): TO-BE-IMPLEMENTED
    """

    def __init__(self,
                 directory=None,
                 ext=[".png", "jpeg", ".jpg"],
                 dump=False,
                 from_file=False,
                 from_list=None,
                 working_dir=None,
                 output="file_list.tsv",
                 transforms=None,
                 verbose=False):
        super(ImageDataset, self).__init__()

        if directory is not None:
            # Remove the slash symbol from the directory name for consistency
            self.directory = directory[:-1] if directory.endswith(
                "/") else directory
        else:
            print(
                "*** Use given file list for dataset, follow the docstring format.***"
            )
        self.extensions = ext if type(ext) is list else [ext]
        self.dump = dump
        self.from_file = from_file

        # Working directory for the data
        if working_dir is not None:
            self.working_dir = working_dir[:-1] if working_dir.endswith(
                "/") else working_dir

        self.output = output
        # [worked with specified transformations] To-Be further implemented
        self.transforms = transforms
        # Get file list from given path or load from given list
        self.file_list = self._get_filelist(
        ) if directory is not None else from_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Fetch label and file path
        file_path, target = self.file_list[index]
        # Load image from given path and convert to torch.Tensor
        image = Image.open(file_path)
        image = self.transforms(image)

        return (image, target)

    def _get_filelist(self):
        if self.from_file:
            # Load all data information from file
            with open(directory, "r") as f:
                tmp = [itr.split("\t") for itr in f.read().splitlines()]

            # Add working directory to the path if working_dir is given
            prefix = "" if self.working_dir is None else self.working_dir + "/"
            tmp = [[itr[0] + prefix, int(itr[1])] for itr in tmp]
            # print("{:d} files contain in the file list".format(len(tmp)))
        else:
            # Get first level of the directory (a.k.a. labels)
            self.labels = os.listdir(self.directory)
            self.n_classes = len(self.labels)

            tmp = list()
            # Get full path name of the files
            for idx, itr_label in enumerate(self.labels):
                # Full path name of current directory
                current_dir = "{:s}/{:s}".format(self.directory, itr_label)
                # Get all files
                all_files = os.listdir(current_dir)

                # Filter out the files with unwanted extensions
                files = list(
                    filter(lambda x: os.path.splitext(x)[1] in self.extensions,
                           all_files))
                # Add full path to the file
                files = [("{:s}/{:s}".format(current_dir, itr), idx)
                         for itr in files]
                # Add all files to the list
                tmp.extend(files)

        # Dump file list to tsv file, format: [PATH_TO_FILE] \t [LABEL]
        if self.dump:
            df = pd.DataFrame(tmp, columns=["Path", "Label"])
            df.to_csv(self.output, sep="\t", header=True, index=False)

        return tmp
