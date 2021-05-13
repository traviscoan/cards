'''A set of helper functions to average the probabilities across
   two models and predict the relevant class.'''

import numpy as np


def predict_claim(probs, labels):
    '''
    Predicts claim based on the maximum probability and returns the
    correct label.

    Args:
        probs (numpy array): An array of probabilities
        labels (list): A list of labels
    
    Returns:
        list: A list of predicted label strings
    '''
    preds_idx = np.argmax(probs, axis=1)
    return [labels[i] for i in preds_idx]


def estimate_ensemble(probs1, probs2, le):
    '''
    Takes the estimated probabilities for 2 models and estimates
    the ensemble classifer by taking a simple average.

    Args:
        probs1 (NumPy array): estimated probabilities from model 1
        probs2 (NumPy array): estimated probabilities from model 2
        le (LabelEncoder object): The scikit-learn label encoder
    
    Returns:
        NumPy array: Averaged class probabilities
    '''
    avg_probs = (probs1 + probs2) / 2
    return predict_claim(avg_probs, le.classes_)

