#!/usr/bin/env python3

"""
Helper script to read a Q table from a .pkl file and print it in human-readable format. Does not mutate data.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple


def read_q_table(path: Path) -> Tuple[int, Dict[str, Dict[str, float]]]:
    """
    Reads a pickled Q-table from the given file.
    :param path: The file to read the Q-table from
    :return: The reconstructed Q-table
    """
    with path.open("rb") as file:
        q_table = pickle.load(file)
    return q_table


def print_q_table(q_table: Tuple[int, Dict[str, Dict[str, float]]], path: Path):
    """
    Prints the given Q-table in a human-readable format to the console (or a file if piped).
    :param q_table: The Q-table to print
    :param path: The path the Q-table was retrieved from (to be printed for identification)
    """
    # create output string, starting w/ path and episode num
    out = "Q-table at %s (episode %d):\n" % (path, q_table[0])

    # separate data
    q_table_data = q_table[1]

    # add different states as rows
    for state_key in q_table_data.keys():
        # create row w/ state key
        row = "%s:\t( " % state_key

        # add action to row
        for action_key in q_table_data[state_key].keys():
            row += "%s:%.2f " % (action_key, q_table_data[state_key][action_key])

        # close row and add to output
        row += ")"
        out += "%s\n" % row

    # print output string
    print(out)


def main(path: Path):
    """
    Main function. Has the script read a Q-table at the given path using Pickle and prints it in human-readable format.
    :param path: The file oath to read the Q-table from
    """
    # load and print Q-table
    try:
        q_table = read_q_table(path)
        print_q_table(q_table, path)
    except IOError as e:
        # log exception
        print("Could not read Q table from file - see error below")
        raise e
    except pickle.UnpicklingError as e:
        # log exception
        print("Could not read Q table from file - see error below")
        raise e


# run on exec
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(prog="read_q_table.py",
                                     description="Helper script to read a Q table from a .pkl file and print it in"
                                                 " human-readable format. Does not mutate data.")
    parser.add_argument("path", type=str, help="Filepath for the Q-table")
    args = parser.parse_args()

    # pass to main
    main(Path(args.path))
