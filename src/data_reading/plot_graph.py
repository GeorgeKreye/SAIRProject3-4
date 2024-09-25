#!/usr/bin/env python3

"""
Helper script used to visualize learning progress.
"""
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union


def load_graph(path: Path) -> Dict[str, Union[int, List[Tuple[int, float]], str]]:
    """
    Loads learning graph data from a file using pickle.
    :param path: The path to load graph data from
    :return: The loaded graph data
    """
    with path.open("rb") as file:
        raw = pickle.load(file)
    if raw['type'] not in ['cr', 'rc', 'lc', 'fc']:  # assert graph type is readable
        raise ValueError("Bad input data - invalid graph type key \'%s\' (should be \'cr\', \'rc\', \'lc\', or "
                         "\'fc\')" % raw['type'])
    return raw


def plot_graph(graph, verbose: bool = False) -> None:
    """
    Plots the learning graph as a line graph.
    :param graph: The graph data to plot
    :param verbose: Whether to print additional information to console
    """
    # get graph type and data
    graph_type: str = graph['type']
    graph_data: List[Tuple[int, float]] = graph['data']

    # split into x and y axes
    graph_x = [point[0] for point in graph_data]
    graph_y = [point[1] for point in graph_data]
    if verbose:
        # print all datapoints to console
        for i in range(0, len(graph_data)):
            print(f"({graph_x[i]},{graph_y[i]})")

    # add data
    plt.plot(graph_x, graph_y)

    # set axis labels
    plt.xlabel("Episode")
    if graph_type == 'cr':
        plt.ylabel("Cumulative reward")
    else:
        plt.ylabel("% of correct choices")

    # set graph title based on type key
    if graph_type == 'cr':
        plt.title("Cumulative reward over time")
    elif graph_type == 'rc':
        plt.title("Learning rate for right close scenario")
    elif graph_type == 'lc':
        plt.title("Learning rate for left close scenario")
    elif graph_type == 'fc':
        plt.title("Learning rate for forward close scenario")
    else:
        raise ValueError("Bad input data - invalid graph type key \'%s\' (should be \'cr\', \'rc\', \'lc\', or "
                         "\'fc\')" % graph_type)

    # show
    plt.show()


def main(path: str, verbose: bool = False):
    """
    Main function. Loads learning graph data from the given file path and plots it as a line graph.
    :param path: The path to retrieve learning graph data from
    :param verbose: Whether to print additional information to console
    """
    # load data
    try:
        graph_data = load_graph(Path(path))
    except IOError as e:
        print("Could not load graph file; see error")
        raise e
    except pickle.UnpicklingError as e:
        print("Could not load graph file; see error")
        raise e

    # plot data
    plot_graph(graph_data, verbose)


# run main on exec
if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(prog="plot_graph.py",
                                     description="Creates a line plot from graph data.")
    parser.add_argument("path", type=str, help="The path to load graph data from")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Whether to print additional information to console")
    args = parser.parse_args()

    # pass to main
    main(args.path, args.verbose)
