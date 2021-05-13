'''Download the CARDS training, validation, and testing sets from
   http://socialanalytics.ex.ac.uk/cards/data.zip'''

import os
import requests
import zipfile
basepath = os.path.dirname(os.path.realpath(__file__))


def download_data(chunk_size=128):
    '''Downloads CARDS data. See the README for more information on the
       folders included in the zip.'''
    r = requests.get('http://socialanalytics.ex.ac.uk/cards/data.zip', stream=True)
    with open(f'{basepath}/data.zip', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def unzip():
    '''Unzip the data.zip the file in the root director.'''
    with zipfile.ZipFile(f'{basepath}/data.zip', 'r') as f:
        f.extractall(basepath)


if __name__ == "__main__":
    download_data()
    unzip()
    os.remove(f'{basepath}/data.zip')

# END