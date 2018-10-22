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
        from_file(bool): Load data from file. Format: [PATH_TO_FILE] \t [LABEL]
        preload(list): If given, the module assumes the list has the format
            [PATH_TO_FILE] \t [LABEL] in each entry.
        working_dir(str): Working directory for the data, to support data stored
            in different place with the file list.
        transforms(torchvision.transform): Customized transformation option.
            [TO-DOs] Support multiple methods.
        verbose(bool): TO-BE-IMPLEMENTED
    """

    def __init__(self,
                 directory=None,
                 ext=[".png", "jpeg", ".jpg"],
                 from_file=False,
                 preload=None,
                 working_dir=None,
                 transforms=None,
                 verbose=False):
        super(ImageDataset, self).__init__()

        assert os.path.isdir(directory) or from_file or isinstance(
            preload, list)
        assert ext is not None

        # Tree structure ordering or data.
        #    [Template]             [Example]
        # [Top]                 # Animals/
        # |- Type_0             # |-cat/
        #    |- file_0          #   |--cat01.jpg
        #    |- file_1          #   |--cat02.jpg
        #    |- ...             #   |-- ...
        # |- Type_1             # |-dog/
        #    |- file_0          #   |--dog/dog01.jpg
        #    |- ...             #   |--...
        # |- ...                # |- ...
        if directory is not None and directory.endswith("/"):
            # Remove the slash symbol from the directory name for consistency
            directory = directory[:-1]
        self.directory = directory

        # The set of file extensions that are considered as valid files
        self.ext = ext if isinstance(ext, list) else [ext]

        # Example of input file format (use tab as separator)
        #    [PATH]          [LABEL]
        # cat/cat01.jpg         0
        # cat/cat02.jpg         0
        # cat/...
        # dog/dog01.jpg         1
        # dog/dog02.jpg         1
        # dog/...
        # bird/bird01.jpg       2
        # bird/...
        # ...
        self.from_file = from_file

        # Working directory for the data
        # (given if the paths in the file is local directory)
        # Example: "Animals/" is locate at "../data"
        # then working_dir should be ../data
        if working_dir is not None and working_dir.endswith("/"):
            working_dir = working_dir[:-1]
        self.working_dir = working_dir

        # [worked with specified transformations] To-Be-Implemented
        self.transforms = transforms

        # If preload is given, check its validity and use it
        if isinstance(preload, list):
            for itr in preload:
                assert os.path.isfile(itr[0]) and isinstance(itr[1], int)
            self._list = preload
        # Get file list from given path or load from given list
        else:
            self._list = self._get_filelist()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        # Fetch label and file path
        _path, target = self._list[index]
        # Load image from given path and convert to torch.Tensor
        image = Image.open(_path)
        if self.transforms is not None:
            image = self.transforms(image)

        return (image, target)

    def _get_filelist(self):
        if self.from_file:
            # Load all data information from file
            with open(directory, "r") as f:
                tmp = [itr.split("\t") for itr in f.read().splitlines()]

            # Add working directory to the path if working_dir is given
            prefix = "" if self.working_dir is None else self.working_dir + "/"
            tmp = [["{}{}".format(prefix, itr[0]), int(itr[1])] for itr in tmp]
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
                _all = os.listdir(current_dir)

                # Filter out the files with unwanted extensions
                files = list(
                    filter(lambda x: os.path.splitext(x)[1] in self.ext, _all))
                # Add full path to the file
                files = [("{:s}/{:s}".format(current_dir, itr), idx)
                         for itr in files]
                # Add all files to the list
                tmp.extend(files)

        return tmp
