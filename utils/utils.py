# Utility functions for CARDS project
import csv
import json


def flatten(list_of_lists):
    '''
    Takes a list of lists and returns the flattned version.

    :param list_of_lists: A list of lists to flatten
    :return: Flattened list
    '''

    return [item for sublist in list_of_lists for item in sublist]


def read_csv(path_):
    '''
    Takes an absolute path to a CSV file and returns file as
    a Python list.

    :param path_: Absolute path to CSV file.
    :return: File as a list of lists.
    '''

    with open(path_, 'r', encoding='utf-8', errors='replace') as csvfile:
        csvreader = csv.reader(csvfile)
        return [row for row in csvreader]


def write_json(path_, content_):
    '''
    Takes a path and list of dictionaries and writes a pretty, POSIX
    compatiable JSON file.

    :param path_: Path to file where JSON should be written
    :return: None
    '''

    with open(path_, 'w') as f:
        json.dump(content_, f, indent=4, separators=(',', ': '), sort_keys=True)
        # add trailing newline for POSIX compatibility
        f.write('\n')


def drop_duplicates(seq):
    '''
    Takes a list and drops duplicates in place.

    :param seq: Iterable
    :return: List without duplicates
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

