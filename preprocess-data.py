import json
import multiprocessing as mp
import re
import sqlite3
from timeit import default_timer as timer

import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
            4. Lemmatization

        variety - format of date (AmE - american type, BrE - british format)
        user_abbrevs - dict of user abbreviations mappings (from normalise package)
        n_jobs - parallel jobs to run

        Source: https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a
        """
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs
        self.regex_html_tag_stripper = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        self.regex_chars_only = re.compile('[^a-zA-Z]')
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

        stemmer = nltk.PorterStemmer()
        self._stem_tokens = np.vectorize(stemmer.stem)

        lemmatizer = nltk.WordNetLemmatizer()
        self._lemmatize_tokens = np.vectorize(lemmatizer.lemmatize)

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        # Strip HTML tags
        cleaned_text = re.sub(self.regex_html_tag_stripper, '', text)

        # Remove non-character letters.
        cleaned_text = re.sub(self.regex_chars_only, ' ', cleaned_text)

        # Create tokens
        tokens = word_tokenize(cleaned_text.lower())

        # Remove inflections of words
        # tokens = self._stem_tokens(tokens)
        tokens = self._lemmatize_tokens(tokens)

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        return tokens


def get_preprocessed_data():
    print('Downloading NLTK resources')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    print('Conneting to SQLite')
    conn = sqlite3.connect("data/raw/database.sqlite")
    df_reviews = pd.read_sql_query("SELECT * FROM Reviews", conn)

    print(f'Dataset contains {df_reviews.shape[0]:n} reviews.')

    df_data = df_reviews[['Text', 'Score']].copy()

    df_data.drop_duplicates(keep='first', inplace=True)
    print(f'Dropped duplicated. Number of reviews: {df_data.shape[0]:n}.')

    df_data['Sentiment'] = df_data['Score'].map(lambda score: 1 if score > 3 else 0)

    df_data = df_data[:1000]  # TODO: Remove this!

    X = df_data['Text']
    y = df_data['Sentiment']

    print('Preprocessing data')
    preprocessor = TextPreprocessor(n_jobs=4)
    start_time = timer()
    X = preprocessor.transform(X)
    end_time = timer()
    duration_secs = end_time - start_time
    print(f'Preprocessing time: {duration_secs:0.2f} secs.')

    return X, y


def docs_to_indices(docs, max_doc_len, vocab):
    N = docs.shape[0]

    docs_indices = np.zeros((N, max_doc_len))
    for i, doc in enumerate(docs):
        # Reserve index 0 for EOF, and index 1 for the UKNOWN token.
        indices = vocab.doc2idx(doc, unknown_word_index=-1)
        indices = [[idx + 2 for idx in indices]]
        padded_indices = pad_sequences(indices, maxlen=max_doc_len, value=0, padding='post')
        docs_indices[i] = padded_indices

    return docs_indices


def run():
    X, y = get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42,
    )

    print(X_train)

    print(f'Number of training samples: {X_train.shape[0]}. Number of test samples: {X_test.shape[0]}')

    max_doc_len = X_train.map(lambda tokens: len(tokens)).max()
    print(f'Max document length: {max_doc_len}')

    train_docs = X_train.to_numpy()

    vocab = Dictionary()
    vocab.add_documents(train_docs)
    vocab.save('data/processed/train-vocab-gensim-dict')

    print(vocab)

    vocab_size = len(vocab) + 2
    embed_dim = 300
    print(f'Vocab size: {vocab_size}')

    print('Converting training texts to indices...')
    X_train_indices = docs_to_indices(X_train, max_doc_len, vocab)

    print(' - Persisting to disk...')
    np.savez_compressed('data/processed/train.npz', X=X_train_indices, y=y_train.to_numpy())

    print('Converting tests texts to indices...')
    X_test_indices = docs_to_indices(X_test, max_doc_len, vocab)

    print(' - Persisting to disk...')
    np.savez_compressed('data/processed/test.npz', X=X_test_indices, y=y_test.to_numpy())

    data_info = {
        'vocab_size': vocab_size,
        'max_doc_len': int(max_doc_len),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

    with open('data/processed/data-info.json', 'w') as f:
        json.dump(data_info, f, indent=2)

    print('Done')


if __name__ == '__main__':
    run()
