#!/usr/bin/env python3

"""
Helper script that finds the best episode based on graph data. Useful for locating the best Q-table generated during
Q-learning training.
"""

# imports
import argparse
import pickle
import os
from pathlib import Path
from typing import Dict, Union, List, Tuple

Graph = Dict[str, Union[int, List[Tuple[int, float]], str]]
"""Dictionary for containing graph dataset"""

GraphData = List[Tuple[int, float]]
"""List of graph datapoints"""

BestDict = Dict[str, Union[Tuple[int, float], Dict[str, Union[int, float]]]]
"""Dictionary for containing best search results"""


def _load_graph(path: Path) -> Graph:
    """
    Loads graph data via Pickle.
    :param path: The path to load graph data from
    :return: The loaded graph data
    :raise IOError: If opening the file from path is not successful
    :raise pickle.UnpicklingError: If graph data cannot be loaded from the file
    """
    # get graph data
    with path.open("rb") as file:
        return pickle.load(file)


def load_graphs(verbose: bool = False) -> Tuple[Graph, Graph, Graph]:
    """
    Loads the graph datasets (right-close, left-close, and forward-close) to be used via Pickle.
    :param verbose: Whether to log additional information to console
    :return: The list of loaded graph datasets
    :raise IOError: If opening a file from path is not successful
    :raise pickle.UnpicklingError: If graph data cannot be loaded from a file
    :raise ValueError: If graph lengths do not match
    """
    # load right-close graph
    if verbose:
        print("Loading right-close graph dataset")
    try:
        rc = _load_graph(Path(os.path.join("graphs", "part2_rc_learning_graph.pkl")))
    except IOError as e:
        print("Could not load right-close graph data: see error")
        raise e
    except pickle.UnpicklingError as e:
        print("Could not load right-close graph data: see error")
        raise e

    # load left-close graph
    if verbose:
        print("Loading left-close graph dataset")
    try:
        lc = _load_graph(Path(os.path.join("graphs", "part2_lc_learning_graph.pkl")))
    except IOError as e:
        print("Could not load left-close graph data: see error")
        raise e
    except pickle.UnpicklingError as e:
        print("Could not load left-close graph data: see error")
        raise e

    # load front-close graph
    if verbose:
        print("Loading front-close graph dataset")
    try:
        fc = _load_graph(Path(os.path.join("graphs", "part2_fc_learning_graph.pkl")))
    except IOError as e:
        print("Could not load front-close graph data: see error")
        raise e
    except pickle.UnpicklingError as e:
        print("Could not load front-close graph data: see error")
        raise e

    # assert graph lengths match
    if rc['iter'] != lc['iter'] != fc['iter']:
        raise ValueError("Graph lengths do not match")

    # return tuple of datasets
    return rc, lc, fc


def _find_individual_best_episode(graph: Graph, min_ep: int = 0) -> Tuple[int, float]:
    """
    Finds the best episode for the given graph by finding the highest success percentage
    :param graph: The graph dataset to use
    :param min_ep: The episode to begin search at; useful for skipping early outliers
    :return: The best episode for the dataset and its success percentage
    :raise ValueError: If the graph is empty
    """
    # get graph length (# of iterations)
    graph_len = graph['iter']
    if graph_len <= 0:
        raise ValueError("No graph data to use - graph is empty")

    # iterate through graph to find best
    data = graph['data']
    best_ep = 0
    best_ep_value = 0.0
    for i in range(min_ep, graph_len):
        # get datapoint
        (ep, ep_value) = data[i]

        # check if best so far
        if ep_value >= best_ep_value:
            # set as best
            best_ep = ep
            best_ep_value = ep_value

    # return best
    return best_ep, best_ep_value


def _average_success_percentage(episode_index: int, graphs: Tuple[GraphData, GraphData, GraphData]) -> float:
    """
    Calculates the average success percentage for a given episode using the given graphs
    :param episode_index: The index of the episode to obtain the average success percentage of
    :param graphs: The graphs to use for calculating the average success percentage
    :return: The average success percentage for the given episode
    """
    # get individual success percentages
    rc_sp = graphs[0][episode_index][1]  # right-close
    lc_sp = graphs[1][episode_index][1]  # left-close
    fc_sp = graphs[2][episode_index][1]  # front-close

    # calculate average
    avg = sum([rc_sp, lc_sp, fc_sp]) / 3

    # return
    return avg


def _find_overall_best_episode(graphs: Tuple[Graph, Graph, Graph], min_ep: int = 0) -> Dict[str, Union[int, float]]:
    """
    Finds the overall best episode by averaging each episode's success percentage across all datasets and
    comparing them.
    :param graphs: The graph datasets to use
    :param min_ep The episode to begin search at; useful for skipping early outliers
    :return: The overall best episode, alongside its average success percentage and its individual success percentages;
     stored in a dictionary with the keys 'ep', 'overall', 'rc', 'lc', and 'fc'
    """
    # get graph length (# of iterations)
    graph_len = graphs[0]['iter']
    if graph_len <= 0:
        raise ValueError("No graph data to use - graphs are empty")

    # iterate through graphs to find overall best
    data = (graphs[0]['data'], graphs[1]['data'], graphs[2]['data'])
    best_ep = 0
    best_ep_avg = 0.0
    best_ep_rc = 0.0
    best_ep_lc = 0.0
    best_ep_fc = 0.0
    for i in range(min_ep, graph_len):
        # get episode
        ep = data[0][i][0]

        # get average success percentage
        avg_success_percentage = _average_success_percentage(i, data)

        # check if best so far
        if avg_success_percentage > best_ep_avg:
            # set as best
            best_ep = ep
            best_ep_avg = avg_success_percentage

            # get individual values
            best_ep_rc = data[0][i][1]  # right-close
            best_ep_lc = data[1][i][1]  # left-close
            best_ep_fc = data[2][i][1]  # forward-close

    # return result dict
    return {'ep': best_ep, 'overall': best_ep_avg, 'rc': best_ep_rc, 'lc': best_ep_lc, 'fc': best_ep_fc}


def find_best_episode(graphs: Tuple[Graph, Graph, Graph],
                      min_ep: int = 0, verbose: bool = False) -> BestDict:
    """
    Using the given graph datasets, finds the best episode overall and for each dataset using their success percentages.
    :param graphs: The graph datasets to use in finding the best episode
    :param min_ep: The episode to begin search at; useful for skipping early outliers
    :param verbose: Whether to log additional information to console
    :return: The numerical index for the best episode overall, as well as the best episode for each graph dataset
     individually; stored as a dictionary with the keys 'overall', 'rc', 'lc', and 'fc', alongside their respective
     success percentages
    """
    # get best overall
    if verbose:
        print('Finding overall best episode')
    overall = _find_overall_best_episode(graphs, min_ep)

    # get best for right-close, left-close, and front-close individually
    # right-close
    if verbose:
        print('Finding best episode for right-close scenario')
    rc = _find_individual_best_episode(graphs[0], min_ep)
    # left-close
    if verbose:
        print('Finding best episode for left-close scenario')
    lc = _find_individual_best_episode(graphs[1], min_ep)
    # front-close
    if verbose:
        print('Finding best episode for front-close scenario')
    fc = _find_individual_best_episode(graphs[2], min_ep)

    # return
    return {'overall': overall, 'rc': rc, 'lc': lc, 'fc': fc}


def print_bests(bests: BestDict):
    """
    Prints the best episode overall and for each graph.
    :param bests:
    """
    # separate dict
    overall = bests['overall']
    rc = bests['rc']
    lc = bests['lc']
    fc = bests['fc']

    # create individual output strings
    overall_str = (f"Best overall:\t{overall['ep']} ({overall['overall']:.2f} - RC={overall['rc']:.2f} "
                   f"LC={overall['lc']:.2f} FC={overall['fc']:.2f})")  # overall
    rc_str = f"Best for right-close scenario:\t{rc[0]} ({rc[1]:.2f})"  # right-close
    lc_str = f"Best for left-close scenario:\t{lc[0]} ({lc[1]:.2f})"  # left-close
    fc_str = f"Best for front-close scenario:\t{fc[0]} ({fc[1]:.2f})"  # front-close

    # combine output strings
    out = "%s\n%s\n%s\n%s" % (overall_str, rc_str, lc_str, fc_str)

    # print output string
    print(out)


def main(min_episode: int = 0, verbose: bool = False):
    """
    Main function. Loads graph datasets, finds the best episode overall and for each dataset,
    and prints them to console.
    :param min_episode: The episode to begin search at; useful for skipping early outliers
    :param verbose: Whether to log additional information to console
    """
    # load graph datasets
    if verbose:
        print("Loading graph datasets")
    graphs = load_graphs()

    # find best episode (overall & individual)
    if verbose:
        print("Calculating best episode, this may take a while")
    bests = find_best_episode(graphs, min_episode, verbose)

    # print
    print_bests(bests)


# run main on exec
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog="find_best_episode.py",
                                     description="Helper script that finds the best episode based on graph data. "
                                                 "Useful for locating the best Q-table generated during Q-learning "
                                                 "training.")
    parser.add_argument("-mi", "--min_episode", type=int, default=0,
                        help="The episode to start searching from (helpful for skipping early outliers)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Log additional data to console")
    args = parser.parse_args()

    # pass to main
    main(args.min_episode, args.verbose)
