import subprocess
import re
import os
from multiprocessing import Pool, cpu_count
from itertools import chain
import numpy as np


class GPUAgent(object):
    """docstring for GPUAgent
    An GPU utilities agent (Docs to be completed)
        Args:
            threshold(int or float): Threshold of available gpus.
            mapping(list): The mapping index of GPUs between information in
                nvidia-smi and actual CUDA_VISIBLE_DEVICES index should be.
                Length should be same as number of gpus seen in nvidia-smi
                e.g. true mapping [1, 2, 3, 4, 0, 5, 6, 7]
            criterion(str): Criterion of which gpu has higher availability.
                [None, "memory", "usage"]
            cmd(str): Custom command for getting GPU info.
    """

    def __init__(self,
                 threshold=0.30,
                 mapping=None,
                 criterion="default",
                 cmd=None,
                 mem_grep=None):
        super(GPUAgent, self).__init__()
        self.threshold = threshold
        self.mapping = mapping
        self.criterion = criterion
        self.cmd = "nvidia-smi" if cmd is None else cmd
        self.mem_grep = "%" if mem_grep is None else mem_grep

    # Set agent attributes
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_cmd(self, cmd):
        self.cmd = cmd

    # Private methods
    def _parse_command(self, cmd, grep=None, opt=None):
        """
        More to be implemented
        Args:
            cmd(str): Command for getting gpu monitoring information.
            grep(str): Grep token for the command.
            opt(str): Options for the command.
        """
        if grep is not None:
            _cmd = "{} | grep {}".format(cmd, grep)
        else:
            _cmd = "{} {}".format(cmd, opt)
        return _cmd

    def _driver_version(self, cmd=None, grep="Driver"):
        # Parse command
        cmd = self.cmd if cmd is None else cmd

        # Full command
        _cmd = self._parse_command(cmd, grep=grep)
        query = self._query_from_console(cmd=_cmd)[0]
        _ver = float(re.findall("Driver Version: (\d+.\d+)", query)[0])
        print("Driver Version: {}".format(_ver))
        return _ver

    def _query_from_console(self, cmd, redundant=None):
        """
        """
        # Get output
        _query = subprocess.check_output(cmd, shell=True)
        _query = _query.decode("ascii").split("\n")[:redundant]

        return _query

    # Public methods
    def get_info(self, cmd=None, opt=None):
        # Parse command
        cmd = self.cmd if cmd is None else cmd
        opt = "-L" if opt is None else opt

        _cmd = self._parse_command(cmd, opt=opt)
        query = self._query_from_console(_cmd, redundant=-1)
        # Example:
        # GPU 0: TITAN X (Pascal) (UUID: GPU-xxxxxxxxxxxxxxxxxxxx)
        names = [re.findall(": (.*?) \(", itr)[0] for itr in query]

        # show info
        print("GPUs: ")
        print(" ID | NAME")
        for idx, itr in enumerate(names):
            print(" {:02d} | {:s}".format(idx, itr))

        return names

    def get_usage(self):
        """
        """
        # Parse threshold
        if self.threshold <= 1:
            info = "{:2.2f}%".format(100 * self.threshold)
        else:
            info = "{:5d}MiB".format(self.threshold)

        cmd = self._parse_command(self.cmd, grep=self.mem_grep)
        # Get all available gpu usage information
        query = self._query_from_console(cmd, redundant=-1)
        #
        if self.mapping is not None:
            assert len(self.mapping) == len(query)
        print("Number of available GPUs: {}".format(len(query)))
        print("Threshold: {:s}".format(info))

        # Get information from the query
        gpu = [[int(itr) for itr in re.findall("(\d+)MiB", i)] for i in query]
        gpu = [[(i[1] - i[0]), 100. * (i[1] - i[0]) / i[1]] for i in gpu]

        # Check availability
        # Note: Higher availability first, not memory size
        # e.g.
        #   GPU 0:  8000MiB /  8192MiB (99%)
        #   GPU 1: 10800MiB / 12288MiB (88%)
        # Decision: GPU 0
        #   GPU 0 has higher percentage available (less occupied)
        #   Even if GPU 1 has larger free memory
        if self.threshold <= 1:
            available = [(itr[1] > self.threshold * 100.) for itr in gpu]
            free = [itr[1] for idx, itr in enumerate(gpu) if available[idx]]
        else:
            available = [(itr[0] > self.threshold) for itr in gpu]
            free = [itr[0] for idx, itr in enumerate(gpu) if available[idx]]

        assert (type(self.criterion) == str) or (self.criterion is None)
        if self.criterion.lower() == "memory":
            print("Overriding decision criterion: {} first".format(
                self.criterion))
            free = [itr[0] for idx, itr in enumerate(gpu) if available[idx]]
        elif self.criterion.lower() == "usage":
            print("Overriding decision criterion: {} first".format(
                self.criterion))
            free = [itr[1] for idx, itr in enumerate(gpu) if available[idx]]
        else:
            # Default: according to the type of threshold
            print("Use default criterion for choosing GPU")

        # Show info
        print("=========================================")
        print("| GPU ID |     Free Memory      |   V   |")
        for idx, itr in enumerate(gpu):
            print("|   {:02d}   | {:5d} MiB ({:2.2f}%) \t| {}\t|".format(
                idx, itr[0], itr[1], available[idx]))
        print("=========================================")

        # Get all GPUs that are available w.r.t the threshold
        usable = [idx for idx, a in enumerate(available) if a]
        # Sort GPU ID by the most available one
        usable = [
            x for _, x in sorted(
                zip(free, usable), key=lambda pair: pair[0], reverse=True)
        ]

        # Remap the GPU IDs if the infomation in nvidia-smi is differ
        # from actual IDs
        if self.mapping is not None:
            usable = [self.mapping[itr] for itr in usable]
        print("Available GPU(s): {}".format(usable))

        self.usable = usable
        return usable

    def get_most_available(self):
        candidates = self.get_usage()
        return candidates[0]

    def set_gpu(self, gpu_id=None):
        gpu_id = self.get_most_available() if gpu_id is None else gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


class cmdAutoAgent(object):
    """docstring for cmdAutoAgent"""

    def __init__(self, arg):
        super(cmdAutoAgent, self).__init__()
        self.arg = arg


class ParallelAgent(object):
    """docstring for ParallelAgent
    """

    def __init__(self, n_process=None):
        super(ParallelAgent, self).__init__()
        self.arg = arg
        self.n_cores = cpu_count()
        self.n_process = (self.n_cores // 2) if n_process == None else n_process

    def split_data(self, data, n_splits):
        """
        Split data to minibatches with last batch may be larger or smaller.
        Arguments:
            data(ndarray): Array of data.
            n_splits(int): Number of slices to separate the data.
        Return:
            partitioned_data(list): List of list containing any type of data.
        """
        n_data = len(data) if type(data) is list else data.shape[0]
        # Slice data for each thread
        print(" - Slicing data for threading...")
        print(" - Total number of data: {0}".format(n_data))

        # Calculate number of instance per partition
        n_part = n_data // n_splits
        partitions = list()
        for idx in range(n_splits):
            # Generate indices for each slice
            idx_begin = idx * n_part
            # Last partition may be larger or smaller
            idx_end = (idx + 1) * n_part if idx != n_splits - 1 else None
            # Append to the list
            partitions.append(data[idx_begin:idx_end])
        #
        return partitions

    def generic_threading(self, n_jobs, data, method, param=None, shared=False):
        """
        Generic threading method.
        Arguments:
            n_jobs(int): number of thead to run the target method
            data(ndarray): Data that will be split and distributed to threads.
            method(method object): Threading target method
            param(tuple): Tuple of additional parameters needed for the method.
            shared: (undefined)
        Return:
            result(list of any type): List of return values from the method.
        """
        # Threading settings
        # n_cores = cpu_count()
        # n_process = (n_cores // 2) if n_jobs == None else n_jobs
        print("Number of CPU cores: {:d}".format(self.n_cores))
        print("Number of Threading: {:d}".format(self.n_process))
        #
        thread_data = split_data(data, self.n_process)
        if param is not None:
            thread_data = [itr + param for itr in thread_data]
        else:
            pass
        #
        print(" - Begin threading...")
        # Threading
        with Pool(processes=self.n_process) as p:
            p.starmap(method, thread_data)
        #
        print("\n" * n_process)
        print("All threads completed.")
        return result if not shared else None


if __name__ == "__main__":
    agent = GPUAgent()
    agent.get_usage()
    agent.set_threshold(5000)
    agent.get_usage()
    agent._driver_version()
    agent.get_info()
    print(agent.get_most_available())
