# CARDS: Computer-assisted recognition of (climate change) denial and skepticism

This repository makes available the training data and main code used to train the classifer described in the following [paper](https://osf.io/preprints/socarxiv/crxfm/):

    "Computer-assisted detection and classification of misinformation about climate change" by Travis G. Coan, Constantine Boussalis, John Cook, and Mirjam Nanko.

### Data

The data used in the paper is available [here] (http://socialanalytics.ex.ac.uk/cards/data.zip). You can download the zipfile (32M) directly or you can use the `download.py` script after cloning the repo to automatically download and unzip the data:

`$ python download.py`

The data directory has two subfolders:

* `analysis/`: Data to replicate the main analysis in the paper.
* `training/`:  Data used to train and test the classifer developed in the paper.



### Contact us

While we've aimed to provide a "complete" list of available transcripts, please reach out if we've missed something or if you spot any errors! Contact info:

* Stephane (S.Baele@exeter.ac.uk)
* Travis (T.Coan@exeter.ac.uk)
