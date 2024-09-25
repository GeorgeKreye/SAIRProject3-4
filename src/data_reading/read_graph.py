#!/usr/bin/env python3

"""
Helper script used to print learning graph data in a human-readable format.
"""
import argparse
import pickle
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


def read_graph(graph: Dict[str, Union[int, List[Tuple[int, float]], str]]):
    """
    Converts learning graph data into a human-readable format and prints it to console.
    :param graph: The graph data to read
    :raise ValueError: If the given graph data is bad (i.e., unexpected attributes)
    """
    # separate data sections
    graph_type = graph['type']
    graph_iter = graph['iter']
    if graph_type == 'cr':
        graph_correct = None
    else:
        graph_correct = graph['correct']
    graph_total = graph['total']
    graph_data = graph['data']

    # create empty output string
    out = ""

    # title
    if graph_type == 'cr':
        out += "Cumulative reward"
    elif graph_type == 'rc':
        out += "Right-close learning"
    elif graph_type == 'lc':
        out += "Left-close learning"
    elif graph_type == 'fc':
        out += "Front-close learning"
    else:
        raise ValueError("Bad input data - invalid graph type key \'%s\' (should be \'cr\', \'rc\', \'lc\', or "
                         "\'fc\')" % graph_type)
    out += "graph @ episode %d\n" % graph_iter

    # totals
    if graph_type == 'cr':
        out += "Last cumulative reward: %d" % graph_total
    else:
        if graph_total > 0:
            out += f"Last correct %: {graph_correct / graph_total} ({graph_correct}/{graph_total})\n"
        else:
            out += "Last correct %: 0 (0/0)\n"

    # data
    out += "=BEGIN DATA=\n"
    for point in graph_data:
        out += f"({point[0]}, {point[1]})\n"
    out += "=END DATA="

    # print filled output string
    print(out)


def main(path: str):
    """
    Main function. Loads learning graph data from the given file path and prints it in human-readable format
    to the console.
    :param path: The path to retrieve learning graph data from
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

    # read graph
    read_graph(graph_data)


# run main on exec
if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(prog="read_graph.py",
                                     description="Prints graph data in a human-readable format.")
    parser.add_argument("path", type=str, help="The path to load learning graph data from")
    args = parser.parse_args()

    # pass to main
    main(args.path)
