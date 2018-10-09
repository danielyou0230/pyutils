import subprocess
import re
import os

class GPU_Agent(object):
    """docstring for GPU_Agent
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
    def __init__(self, threshold=0.3, mapping=None, criterion="default", cmd=None):
        super(GPU_Agent, self).__init__()
        self.threshold = threshold
        self.mapping = mapping
        self.criterion = criterion
        self.cmd = cmd if cmd is not None else "nvidia-smi | grep %"

    # Set agent attributes
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_cmd(self, cmd):
        self.cmd = cmd

    def get_gpu_info(self):
        if self.threshold <= 1:
            info = "{:2.2f}%".format(100 * self.threshold)
        else:
            info = "{:5d}MiB".format(self.threshold)
        # Get all available gpu usage information
        output = subprocess.check_output(self.cmd, shell=True)
        output = output.decode("ascii").split("\n")[:-1]
        #
        if self.mapping is not None:
            assert len(self.mapping) == len(output)
        print("Number of available GPUs: {}".format(len(output)))
        print("Threshold: {:s}".format(info))

        # Get information from the output
        gpu = [[int(itr) for itr in re.findall("(\d+)MiB", i)] for i in output]
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
            print("Overriding decision criterion: {} first".format(self.criterion))
            free = [itr[0] for idx, itr in enumerate(gpu) if available[idx]]
        elif self.criterion.lower() == "usage":
            print("Overriding decision criterion: {} first".format(self.criterion))
            free = [itr[1] for idx, itr in enumerate(gpu) if available[idx]]
        else:
            # Default: according to the type of threshold
            print("Use default criterion for choosing GPU")

        # Show info
        print("=========================================")
        print("| GPU ID |     Free Memory      |   V   |")
        for idx, itr in enumerate(gpu):
            print("|   {:02d}   | {:5d} MiB ({:2.2f}%) \t| {}\t|".format(idx, itr[0], itr[1], available[idx]))
        print("=========================================")

        # Get all GPUs that are available w.r.t the threshold
        usable = [idx for idx, a in enumerate(available) if a]
        # Sort GPU ID by the most available one
        usable = [x for _, x in sorted(zip(free, usable), key=lambda pair: pair[0], reverse=True)]

        # Remap the GPU IDs if the infomation in nvidia-smi is differ
        # from actual IDs
        if self.mapping is not None:
            usable = [self.mapping[itr] for itr in usable]
        print("Available GPU(s): {}".format(usable))

        self.usable = usable
        return usable

    def getMostAvailable(self):
        candidate = self.get_gpu_info()
        return candidate[0]

    def set_gpu(self, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

if __name__ == "__main__":
    agent = GPU_Agent()
    agent.get_gpu_info()
    agent.set_threshold(5000)
    agent.get_gpu_info()
