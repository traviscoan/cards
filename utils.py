'''A handful of utility functions.'''

import csv
import json


def flatten(list_of_lists):
    '''
    Takes a list of lists and returns the flattned version.

    Args:
        list_of_lists (list): A list of lists to flatten
    
    Returns: 
        list: The flattened list
    '''

    return [item for sublist in list_of_lists for item in sublist]


def read_csv(path, remove_header=False):
    '''
    Takes an absolute path to a CSV file and returns file as
    a Python list.

    Args:
        path (str): Absolute path to CSV file.
        remove_header (bool): Optionally the first line of the CSV.
    
    Returns: 
        list: The CSV file as a list of lists.
    '''

    with open(path, 'r', encoding='utf-8', errors='replace') as csvfile:
        csvreader = csv.reader(csvfile)
        data_ = [row for row in csvreader]
    
    if remove_header:
        data = data_[1:]
    else:
        data = data_
    
    return data


def write_json(path, content):
    '''
    Takes a path and list of dictionaries and writes a pretty, POSIX
    compatiable JSON file.

    Args:
        path (str): Path to file where JSON should be written.
        content (list): List of dictionaries to write.

    Returns:
        None
    '''

    with open(path, 'w') as f:
        json.dump(content, f, indent=4, separators=(',', ': '), sort_keys=True)
        # add trailing newline for POSIX compatibility
        f.write('\n')


def drop_duplicates(seq):
    '''
    Takes a list and drops duplicates in place.

    Args:
        seq (iterable): Iterable
    
    Returns: 
        list: List without duplicates
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

