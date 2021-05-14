# CARDS: Computer-assisted recognition of (climate change) denial and skepticism

This repository makes available the training data and main code used to train the classifer described in the following [paper](https://osf.io/preprints/socarxiv/crxfm/):

    "Computer-assisted detection and classification of misinformation about climate change" 
    by Travis G. Coan, Constantine Boussalis, John Cook, and Mirjam Nanko.

### Data

The data used in the paper is available [here](http://socialanalytics.ex.ac.uk/cards/data.zip). You can download the zipfile (32M) directly or you can use the `download.py` script after cloning the repo to automatically download and unzip the data:

`python download.py`

The data directory has two subfolders:

* `analysis/`: Data to replicate the main analysis in the paper.
* `training/`:  Data used to train and test the classifer developed in the paper.

### Code

While we imagine that most people will want to download the training data and roll their own classifer, we also provide the code used to train and test the model presented in the paper. To use this code, start by creating and attaching a Python virtual environment:

`python -m venv env`

`source env/bin/activate`

And then pip install the project requirements:

`pip install -r requirements.txt`

Note that we use `spaCy`'s `en_core_web_lg` (778.7M zipped) to tokenize the text, which be downloaded automatically as a requirement. If you want to avoid this, you can remove it from the `requirements.txt` and edit the `preprocess.py` script accordingly.

The code is organized as follows:

* `preprocess.py`: A set of functions to preprocess and tokenize the text.
* `fit/`: As described more fully in the paper, we our model uses a simple ensemble of a logistic classifier and the [RoBERTa](https://arxiv.org/abs/1907.11692) architecture. The [fit](https://github.com/traviscoan/cards/tree/master/fit) directory provides the code used to fit both parts of the model, as well as code to combine the two estimates. See [fit](https://github.com/traviscoan/cards/tree/master/fit) for more information.

### Contact us

While we've aimed to provide a "complete" list of available transcripts, please reach out if we've missed something or if you spot any errors! Contact info:

* Travis (T.Coan@exeter.ac.uk)
