import json
import glob


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


files = sorted(glob.glob('data/full_data/*.json'))

cards_with_probs = []

for file in files:
    with open(file, 'r') as jfile:
        content = json.load(jfile)

    for row in content:
        cards_with_probs.append(row)


write_json('data/full_data/cards_with_probs.json', cards_with_probs)