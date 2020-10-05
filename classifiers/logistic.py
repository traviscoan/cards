"""Fit the optimised logistic classifer to the CARDS corpus."""

import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read training data
df_train = pd.read_json('data/training_data/train.json')

# DOOKIE: reomove need for this preprocessing
df_train['sub_claim_combined'] = df_train['sub_claim_combined'].astype('str')
df_train['sub_claim_combined'] = df_train['sub_claim_combined'].str[:3]
df_train['sub_claim_combined'] = df_train['sub_claim_combined'].str.replace('\.', '_')
cards_train = df_train.to_dict(orient='records')

# Encode labels
claims = [str(row['sub_claim_combined']) for row in cards_train]
le = preprocessing.LabelEncoder()
y = le.fit_transform(claims)

# Vectorize
text = [row['tokens'] for row in cards_train]
vectorizer = TfidfVectorizer(min_df=3,  max_features=None,
                             strip_accents='unicode',
                             ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

X = vectorizer.fit_transform(text)


# Fit final logistic classifier. Hyperparameters tuned via grid search using
#  10-fold cross-validation
clf_logit = LogisticRegression(C=7.96,
                               solver='lbfgs',
                               multi_class='ovr',
                               max_iter=200,
                               class_weight='balanced')

# Fit final logit model
clf_logit.fit(X, y)

# Save model
model = {'clf': clf_logit, 'label_encoder': le, 'vectorizer': vectorizer}

with open('classifiers/fitted_models/logistic.pkl', 'wb') as f:
    pickle.dump(model, f)
