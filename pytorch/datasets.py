import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.misc import imresize


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


class VideoDataset(Dataset):
    """
    docstring for VideoDataset

    Args:
        clipFileList(str): Path to the filelist.
        size(tuple of ints): Size of the input to be reshaped to.
        random(bool): Randomize the starting frame.
        stride(int): Stride of sampling frames.
        colorSpace(str): ["RGB", "BW"]
    """

    def __init__(self,
                 clipFileList,
                 size,
                 n_clip=32,
                 random=True,
                 stride=2,
                 colorSpace="RGB",
                 grayscale=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None):
        super(videoDataset, self).__init__()
        self.cliplist = self.getClipList(clipFileList)  # list()

        # Parameters for video segments
        self.random = random
        self.stride = stride
        self.n_clip = n_clip
        self.size = size
        # Calculate time span of a segment
        self.time_span = stride * n_clip
        # Depth (Channels)
        self.clip_ch = {"RGB": 3, "BW": 1}
        self.channels = self.clip_ch[colorSpace]
        self.grayscale = grayscale
        self.transform = transform
        self.basic_transform = transforms.Compose(
            [transforms.Normalize(mean, std)])

        print("files: {}".format(self.cliplist))

    def __len__(self):
        return len(self.cliplist)

    def __getitem__(self, index):
        return self.readVideo(self.cliplist[index])

    def getClipList(self, filename):
        with open(filename, "r") as f:
            cliplist = f.read().splitlines()

        cliplist = [itr.split(" ") for itr in cliplist]
        return cliplist

    def sampleFrame(self, f):
        # Normal vidoes
        if (f > self.time_span) and self.random:
            begin = np.random.choice(f - self.time_span)
        # Video with frame number less than time span.
        else:
            begin = 0

        # Sample the frame indices
        sampled_idx = [(begin + i * self.stride) for i in range(self.n_clip)]
        sampled_idx = np.array(sampled_idx)

        if f < self.time_span:
            # sampled_idx = [i % f for i in range(self.n_clip)]
            sampled_idx = sampled_idx % f

        return sampled_idx

    def readVideo(self, filename):
        # Open the video file
        cap = cv2.VideoCapture(filename[0])
        # Get Video info
        frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # frames = torch.FloatTensor(self.channels, frameC, frameH, frameW)
        frames = torch.FloatTensor(self.channels, frameC, self.size, self.size)
        failedClip = False
        for f in range(frameC):
            # Capture video from file
            valid, frame = cap.read()
            if valid:
                # Resize frame
                frame = imresize(frame, size=(self.size, self.size))
                frame = torch.from_numpy(frame).float()
                # dimension: HWC -> CHW
                frame = frame.permute(2, 0, 1)

                # Apply transformation on the frame
                frame = self.basic_transform(frame)
                if self.transform is not None:
                    frame = self.transform(frame)

                # Write frame to tensor
                frames[:, f, :, :] = frame

            else:
                print("Skipped: {:s} (at frame {:d})".format(filename, f))
                failedClip = True
                break

        # Randomly sample frames from videos
        sampled_idx = self.sampleFrame(frameC)
        frames = frames[:, sampled_idx, :, :]
        # Set label for the video
        label = int(filename[1])

        return frames, label  # , failedClip
