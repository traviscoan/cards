# CARDS: Computer-assisted recognition of (climate change) denial and skepticism

This repository makes available the training data and main code used to train the classifer described in the following [paper](https://osf.io/preprints/socarxiv/crxfm/):

    "Computer-assisted detection and classification of misinformation about climate change" 
    by Travis G. Coan, Constantine Boussalis, John Cook, and Mirjam Nanko.

## Data

The data used in the paper is available [here](https://drive.google.com/uc?export=download&id=14exmlYCT3-K2byYHFFrShAIYiemJQroi). After unzipping the file, you will find a data directory with two subfolders:

* `analysis/`: Data to replicate the main analysis in the paper.
* `training/`:  Data used to train and test the classifer developed in the paper.

## Code

While we imagine that most people will want to download the training data and roll their own classifer, we also provide the code used to train and test the model presented in the paper. To use this code, start by creating and attaching a Python virtual environment:

`python -m venv env`

`source env/bin/activate`

And then pip install the project requirements:

`pip install -r requirements.txt`

Note that we use `spaCy`'s `en_core_web_lg` (778.7M zipped) to tokenize the text, which be downloaded automatically as a requirement. If you want to avoid this, you can remove it from the `requirements.txt` and edit the `preprocess.py` script accordingly.

### Fitting the RoBERTa-Logistic ensemble classifer

As described more fully in the paper, our model uses a simple ensemble of a logistic classifier and the [RoBERTa](https://arxiv.org/abs/1907.11692) architecture.

#### RoBERTa

We used the fantastic `simpletransformers` library to train, test, and perform inference for the RoBERTa side of our model. For more on how to install the `simpletransformers` library, please see:

https://simpletransformers.ai/docs/installation/

The `roberta/` subdirectory contains the code to fit and evaluate the model. Specifically, directory includes the following 3 Jupyter notebooks which walk you through the process:

* `cards_training.ipynb`: Includes the code (and, importantly, the hyperparameters) used to fit the CARDS model.
* `cards_evaluation.ipynb`: Provides the code to evaluate the model performance on held-out data.
* `cards_inference.ipynb`: Provides code to infer classes in unseen data.

#### Logistic classifer

To run the logistic side of the model, you can use the following process:

```python
from utils import read_csv
import preprocess as pp
from fit.logistic import fit_logistic_classifier

# Read in downloaded training data
data = read_csv("data/training/training.csv", remove_header=True)

# Clean and tokenize. Uses unthreaded spaCy, so it is slow!
tokens = [[pp.tokenize(pp.denoise_text(row[0]), remove_stops=True), row[1]] for row in data]

# Fit the model
model = fit_logistic_classifier(tokens)
```

The model dictionary holds the trained model, vectorizer, and label encoder (`help(fit_logistic_classifer)`).

### Pre-trained model weights

The pre-trained model weights for the RoBERTa (and logistic) model used in the paper are available [here](https://drive.google.com/uc?export=download&id=1cbASuoLNY-kJcm7hUFLTGYzblZFzxaVo). Note that this file is large (3.5G zipped) -- sorry!




