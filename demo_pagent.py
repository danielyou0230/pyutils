import time
import numpy as np
from tqdm import tqdm
from agents import ParallelAgent


# *** Demo Code Section ***
def demo_method(data, a, b):
    """
    Template for writting method to perform parallel operations.
    Arguments a and b can be arbitrary number of parameters of none.

    Arguments:
        data(iterable): Iterable instances.
        a()
        b()
    """
    # Perform single-instance operations/preprocessing here.
    time.sleep(0.25)  # sleep for 0.25s
    _processed = "data {} with params {} {}".format(data, a, b)
    # End of operation

    return _processed


def demo_method_batch(process_idx, data, a, b):
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
        # Perform batch operations/preprocessing here.
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


def main():
    """
    Demonstration on how to deploy ParallelAgent
    """
    # Pre-settings for multiprocessing
    agent = ParallelAgent(n_process=2)

    # Add parameters if needed by the method (either way is fine)
    # (1) create a tuple for those parameter(s) and pass it in run()
    #   only one parameter: (parameter1, )
    #   many parameters :   (parameter1, parameter2, ...)
    param = ("first", "second")
    # (2) set parameter using method add_parameters()
    # agent.add_parameters(param)

    # Create random data (any iterable instance is fine)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                     [13, 14, 15]])

    # Run in parallel
    # "method": Must be given in run().
    results = agent.run(data, param=param, method=demo_method, unpack=True)

    # *** Demo mode only code ***
    print("\n[DEMO] Return list contains:")
    for idx, itr in enumerate(results, 1):
        print(" - from Process {:02d}: {}".format(idx, itr))
    print("\nEntire list looks like this:\n{}".format(results))


if __name__ == "__main__":
    main()
