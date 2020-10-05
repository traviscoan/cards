import json
import pickle
import numpy as np
import pandas as pd
from process.preprocess import denoise_text
from process.preprocess import tokenize
from utils.utils import flatten
from utils.utils import write_json


# Helpers
def _preprocess(text):
    '''Takes raw paragraph text, clean, and tokenises.'''
    text_ = denoise_text(text)
    return ' '.join(flatten(tokenize(text_, remove_stops=True)))


def predict_claim(probs, labels):
    '''
    Predicts claim based on the maximum probability and returns the
    correct label.

    :param probs: array of probabilities
    :param labels: list of labels
    :return:
    '''
    preds_idx = np.argmax(probs, axis=1)
    return [labels[i] for i in preds_idx]


def estimate_ensemble(probs1, probs2, le):
    '''
    Takes the estimated probabilities for 2 models and estimates
    the ensemble classifer by taking a simple average.

    :param probs1: estimated probabilities from model 1
    :param probs2: estimated probabilities from model 2
    :param le: label econder
    :return:
    '''
    avg_probs = (probs1 + probs2) / 2
    return predict_claim(avg_probs, le.classes_)


# Read full data
with open('data/full_data/cards_with_probs.json', 'r') as jfile:
    content = json.load(jfile)

# Tokenize text for logistic classifer
errors = []
for i, row in enumerate(content):
    try:
        row['tokens'] = _preprocess(row['text'])
    except:
        row['tokens'] = None
        errors.append(row)

    if (i % 10000) == 0:
        print("Finished %s" % i)

# Remove missing text
content_clean = [row for row in content if row['tokens'] is not None]

# Write for later use
write_json('data/full_data/cards_paras_probs_tokens.json', content_clean)

# Load logistic
with open('classifiers/fitted_models/logistic.pkl', 'rb') as f:
    logit = pickle.load(f)

# Parse model content
vectorizer = logit['vectorizer']
clf = logit['clf']
le = logit['label_encoder']

# Estimate logit model
tokens = [row['tokens'] for row in content_clean]
X = vectorizer.transform(tokens)
logit_probs = clf.predict_proba(X)

# Extract pre-inferred ULMFiT weights
ulmfit_probs = np.array([row['ulmfit_probs'] for row in content_clean])

# Estimate the ensemble
sub_claim = estimate_ensemble(logit_probs, ulmfit_probs, le)
# sub_claim = le.inverse_transform(sub_claim)

# Write dataset for analysis
dat = []
for i, row in enumerate(content_clean):
    dat.append({'domain': row['domain'],
                'date': row['date'],
                'ctt_status': row['ctt_status'],
                'pid': row['pid'],
                'text': row['text'],
                'claim': sub_claim[i]})

dat = pd.DataFrame(dat)
dat.to_csv('data/full_data/cards_for_analysis.csv')
