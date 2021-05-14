'''A set of functions used to preprocess the CARDS text.'''

import re
from bs4 import BeautifulSoup
import unicodedata
from nltk.corpus import stopwords
import spacy
from utils import flatten

# Initialize globals
import en_core_web_lg
nlp = en_core_web_lg.load()
STOP_WORDS = stopwords.words('english')
POS_TO_REMOVE = set(['PUNCT', 'SPACE'])


def strip_html(text):
    '''
    Strips any remaining HTML from a text string.

    Args:
        text (str): Text to strip
    
    Returns:
        str: Stripped text.
    '''
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    '''
    Removes text between square brackets from a string.

    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text.
    '''
    return re.sub('\[[^]]*\]', '', text)


def remove_non_ascii(text):
    '''
    Removes non-ASCII characters from a text string.

    Args:
        text (str): Text with non-ASCII characters
    
    Returns:
        str: Text with non-ASCII characters.
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def strip_underscores(text):
    '''
    Removes underscores from a text string.

    Args:
        text (str): Text string
    
    Returns:
        str: Text string with underscores.
    '''
    return re.sub(r'_+', ' ', text)


def remove_multiple_spaces(text):
    '''
    Remove multiple spaces from a text string.

    Args:
        text (str): Text string
    
    Returns:
        str: Text string without multiple spaces.
    '''
    return re.sub(r'\s{2,}', ' ', text)


def denoise_text(text):
    '''
    "Denoise" a text string.

    Args:
        text (str): Text string
    
    Returns:
        str: Text string sans noise.
    '''
    text_no_html = strip_html(text)
    text_no_square = remove_between_square_brackets(text_no_html)
    text_no_ascii = remove_non_ascii(text_no_square)
    text_no_under = strip_underscores(text_no_ascii)
    text_no_multiple = remove_multiple_spaces(text_no_under)
    return text_no_multiple.strip()


def tokenize(document, remove_stops=False, lemmatize=False):
    '''
    Takes a document, splits into sentences, and processes
    the sentence (tokenize, remove stopwords, etc.) using
    spacy.

    Args:
        document (str): A document (or text entry) to process.
        remove_stops (bool): Whether or not to remove stopwords.
        lemmatize (bool): Lemmatize tokens.

    Returns:
        tuple: A list of tokens and sentences (tokens, sentences).
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

    return ' '.join(flatten(tokens)).strip()

