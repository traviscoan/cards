"""Fit the optimised logistic classifer to the CARDS corpus."""

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_logistic_classifier(data):
    '''
    Takes training "data" and fits a LogisticRegression classifier.

    Args:
        data (list): A list of lists holding the paragraph text (index 0) 
                     and the claim (index 1).
    
    Returns:
        dict: A dictionary with the following keys:
              "clf" (LogisticRegression): the fitted logistic classifer
              "label_encoder" (LabelEncoder): encoder used for labels.
              "vectorizer" (TfidfVectorizer): Fitted TF-IDF vectorizer.           
    '''
    # Encode labels
    claims = [row[1] for row in data]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(claims)

    # Vectorize
    text = [row[0] for row in data]
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

    return {'clf': clf_logit, 'label_encoder': le, 'vectorizer': vectorizer}


