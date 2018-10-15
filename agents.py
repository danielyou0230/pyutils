import subprocess
import re
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import chain
from tqdm import tqdm
import time
from itertools import chain


class GPUAgent(object):
    """docstring for GPUAgent
    An GPU utilities agent (Docs to be completed)
        Arguments:
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
        Arguments:
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
    """ docstring for ParallelAgent

    Arguments:
        n_process(int): The number parallel processes to run.
        split(method): Customized split data methods. If not give, use
            pre-defined self._split_data() as method to split data to
            all paralled processes.
    """

    def __init__(self, n_process=-1, split=None):
        super(ParallelAgent, self).__init__()
        self.n_cores = cpu_count()
        self.n_process = self.n_cores // 2 if n_process == -1 else n_process

    def _split_data(self, data):
        """
        Split data to partitions with last batch may be larger or smaller.

        Arguments:
            data(ndarray): Array of data.
        Return:
            partitions(tuple): tuple of list containing, use tuple to give
                required parameters to the methods (in order):
                (P_IDX, _data)

                P_IDX: Process index for aligning progress bars.
                _data: Partitioned data for each parallel process.
        """
        n_data = len(data) if type(data) is list else data.shape[0]
        # Calculate number of instance per partition
        n_part = n_data // self.n_process

        # Slice data for each thread
        print(" - Partitioning data for parallelisation...")
        print(" - Total number of instances: {}".format(n_data))

        partitions = list()
        for idx in range(self.n_process):
            # Generate indices for each slice
            begin = idx * n_part
            # Last partition may be larger or smaller
            end = (idx + 1) * n_part if idx != self.n_process - 1 else None
            # Append one part of instances to the list
            partitions.append((idx + 1, data[begin:end]))
        #
        return partitions

    def add_parameters(self, param):
        self.param = param

    def run(self,
            data,
            method=None,
            n_process=None,
            param=None,
            unpack=False,
            demo=False):
        """
        Generic threading method.
        Arguments:
            data(iterable objects): Data that desired to be processed in
                parallel, should be iterable objects like list or np.array.
            method(method object): Threading target method
            n_process(int): The number of process to be run in parallel.
            param(tuple): A tuple of all additional parameters needed for
                the method to be run in parallel.
            unpack(bool): Unpack the result from all process, or flatten,
                which would look like input data
            demo(bool): Demo mode, perform demo methods if asserted.
        Return:
            result(list of any type): List of return values from the method.
        """
        if n_process is not None:
            self.n_process = n_process
        if param is not None:
            self.param = param

        assert (method is not None) or demo
        if demo:
            method = self.demo_method

        # Show multiprocessing settings
        print("Number of CPU cores: {:d}".format(self.n_cores))
        print("Number of Threading: {:d}".format(self.n_process))
        #
        _data = self._split_data(data)
        # Add parameters to each partitions if given
        if self.param is not None:
            _data = [itr + self.param for itr in _data]

        print(" - Begin threading...")
        # Threading
        with Pool(processes=self.n_process) as p:
            results = p.starmap(method, _data)

        # Compensate those lines occupied by those progressbars
        print("\n" * self.n_process)
        print("Multiprocessing completed.")

        # *** Demo mode only code BEGIN ***
        if demo:
            print("[DEMO] Return list contains:")
            for idx, itr in enumerate(results, 1):
                print(" - from Process {:02d}: {}".format(idx, itr))
            print("Entire list looks like this:\n{}".format(results))

        # Returning the results after parallel processing
        if unpack:
            # WILL SUPPORT np.array after testing
            if type(data) == list:
                results = list(chain.from_iterable(results))
            else:
                print("Input is not list, please manually unpack it.")
        # *** Demo mode only code END ***

        # No unpacking, remember to unpack yourself
        else:
            pass

        return results

    def demo_method(self, process_idx, data, a, b):
        """
        """
        # This line is just demonstrating what a process would get unneeded
        print("Process {:02d} got: {}".format(process_idx, data))

        # Follow the following two lines
        desc = "Process {:02d}".format(process_idx)
        result = list()

        # Any processing function you would like to implemented recursively
        # Explanation on passing "position" and "desc":
        #   position(int): Manually aligning those progressbar for easilier
        #           monitoring and interpreting (prevent them from
        #           overlapping each other.)
        #   desc(str): Marking the process's progressbar (shown before the
        #           progressbar.)
        #
        # Illustration:
        #   [desc]
        # Process 01: 100%|████████████████████| 2/2 [00:02<00:00,  1.00s/it]
        # Process 02: 100%|████████████████████| 3/3 [00:03<00:00,  1.00s/it]

        # Follow the above guide for first line
        for itr in tqdm(data, position=process_idx, desc=desc):
            # Do anything you want to perform in parallel.
            time.sleep(0.25)  # sleep for 0.25s
            _processed = "{} working at {}".format(process_idx, itr)

            # Add to the list
            result.append(_processed)

        # Returning the list
        # (After returning, the list that collects all results from all
        # processes is a list of length equal to the number of parallel
        # processes with each element being the individual processed inputs.)
        #
        # results = [result_from_p1, result_from_p2, ...]
        # each result_from_p* would be a list of processed instances
        return result

    def demo(self):
        # Pre-settings for multiprocessing
        self.n_process = 2

        # Add parameters if needed by the method (either way is fine)
        # (1) create a tuple for those parameter(s) and pass it in run()
        #   only one parameter: (parameter1, )
        #   many parameters :   (parameter1, parameter2, ...)
        param = ("first", "second")
        # (2) set parameter using method add_parameters()
        # pagent.add_parameters(param)

        # Create random data (any iterable instance is fine)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

        # Run in parallel
        # "method": Must be given in run().
        # "demo"  : Used to set and demonstrate the usage of this
        #           agent, DO NOT SET IT TO TRUE while deploying it.
        result = pagent.run(data=data, param=param, unpack=True, demo=True)
        #
        # Actual usage should look like this
        # result = pagent.run(data=data, method=SOME_METHOD, param=param)


if __name__ == "__main__":
    agent = GPUAgent()
    agent.get_usage()
    agent.set_threshold(5000)
    agent.get_usage()
    agent._driver_version()
    agent.get_info()
    print(agent.get_most_available())
    pagent = ParallelAgent()
    pagent.demo()
