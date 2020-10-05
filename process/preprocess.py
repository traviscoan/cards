import sys
import os
import argparse
import pandas as pd
import re
from bs4 import BeautifulSoup
import unicodedata
from utils.utils import flatten
from utils.utils import write_json
from nltk.corpus import stopwords
import spacy

# Initialize globals
import en_core_web_lg
nlp = en_core_web_lg.load()
STOP_WORDS = stopwords.words('english')
POS_TO_REMOVE = set(['PUNCT', 'SPACE'])
LOG_EVERY_N = 1000


def standardize(content, codes):
    # Super claim
    merged = content.merge(codes, how='left', left_on='Super_Claim', right_on='Id')
    merged = merged.rename(index=str, columns={"Id": "super_id",
                                               "Claim_Text": "super_label",
                                               "Code": "super_code"})

    # Sub-claim
    merged = merged.merge(codes, how='left', left_on='Sub_Claim', right_on='Id')
    merged = merged.rename(index=str, columns={"Id": "sub_id",
                                               "Claim_Text": "sub_label",
                                               "Code": "sub_code"})

    # Sub-sub-claim
    merged = merged.merge(codes, how='left', left_on='Sub_Sub_Claim', right_on='Id')
    merged = merged.rename(index=str, columns={"Id": "sub_sub_id",
                                               "Claim_Text": "sub_sub_label",
                                               "Code": "sub_sub_code"})

    return merged[['ParagraphId', 'Paragraph_Text', 'phase', 'super_label', 'super_code', 'sub_label',
                   'sub_code', 'sub_sub_label', 'sub_sub_code']]


def make_3digit_code(line):
    if line['sub_sub_label'] != 'No claim':
        code = line['sub_sub_code']
    else:
        if line['sub_label'] != 'No claim':
            code = '%s.0' % line['sub_code']
        else:
            code = '%s.0.0' % line['super_code']

    return code


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def strip_underscores(text):
    return re.sub(r'_+', ' ', text)


def remove_multiple_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_non_ascii(text)
    text = strip_underscores(text)
    text = remove_multiple_spaces(text)
    return text.strip()


def normalize_codes(cards, codes):
    cards = standardize(cards, codes)

    # Handle 'No claim' entries
    cards = cards.fillna('No claim')
    cards = cards.to_dict(orient='records')

    # Change "None" to 0
    for row in cards:
        if row['super_code'] == 'No claim':
            row['super_code'] = '0'
            row['super_label'] = 'No claim'
        if row['sub_code'] == 'No claim':
            row['sub_code'] = '0'
            row['sub_label'] = 'No claim'
        if row['sub_sub_code'] == 'No claim':
            row['sub_sub_code'] = '0'
            row['sub_sub_label'] = 'No claim'

    # Add the normalized, 3 digit code
    for i, row in enumerate(cards):
        row['sub_sub_claim'] = make_3digit_code(row)

    return cards


def tokenize(document, remove_stops=False, lemmatize=False):
    '''
    Takes a document, splits into sentences, and processes
    the sentence (tokenize, remove stopwords, etc.) using
    spacy.

    :param document: A document (or text entry) to process (str)
    :param remove_stops: Whether or not to remove stopwords
    :param lemmatize: Lemmatize tokens

    :return: A tuple with a list of tokens and sentences
             (tokens, sentences)
    '''

    # Process "document" (i.e., paragraphs)
    doc = nlp(document)

    # Tokenize
    tokens = []

    # Should we lemmatize:
    if lemmatize:
        for sentence in doc.sents:
            if remove_stops:
                tokens.append([tok.lemma_ for tok in sentence if tok.pos_ not in POS_TO_REMOVE
                               and tok.lemma_ not in STOP_WORDS and tok.lemma_ != ''])
            else:
                tokens.append([tok.lemma_ for tok in sentence if tok.pos_ not in POS_TO_REMOVE])
    else:
        for sentence in doc.sents:
            if remove_stops:
                tokens.append([tok.text.lower() for tok in sentence if tok.pos_ not in POS_TO_REMOVE
                               and tok.text.lower() not in STOP_WORDS])
            else:
                tokens.append([tok.text.lower() for tok in sentence if tok.pos_ not in POS_TO_REMOVE])

    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cardsdir", type=str,
                        help="The main directory housing the CARDS data and code.")
    parser.add_argument("-c", "--cutoff", type=int,
                        help="Number of tokens considered *too short*.")
    args = parser.parse_args()

    if args.cutoff is None:
        print('Need to specify a cutoff value for short texts.')
        sys.exit()

    # Switch to main directory
    os.chdir(args.cardsdir)

    # Read data
    cards = pd.read_json('data/cards.json')

    # Tranform the claims for readability. First, read map from internal codes
    # to taxonomy codes and then normalize
    print('Normalizing claims for readability...')
    codes = pd.read_csv('data/claims_map.csv')
    cards = normalize_codes(cards, codes)

    # Remove noise
    print('Removing noisy text...')
    for row in cards:
        row['Paragraph_Text'] = denoise_text(row['Paragraph_Text'])

    # Tokenize for shallow learning
    print('We use spacy for tokenization and it is SLOW! Patience is a virtue.')
    for i, row in enumerate(cards):
        # Process paragraph. Tokenization regex only brings back letters
        # and numbers
        tokens = tokenize(row['Paragraph_Text'], remove_stops=True)

        # Append result to dictionary
        row['tokens'] = ' '.join(flatten(tokens)).strip()

        if (i % LOG_EVERY_N) == 0:
            print("Finished %s" % i)

    # Remove really, really short text
    print('Removing short texts (less than %s words).' % args.cutoff)
    cards = [row for row in cards if len(row['tokens'].split(' ')) > args.cutoff]

    # Write standardized text to disk
    print('Writing standardized text to disk.')
    write_json('data/cards_standardized.json', cards)
