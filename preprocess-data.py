import numpy as np
import multiprocessing as mp

import string
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import re

import nltk
from nltk.tokenize import word_tokenize

import sqlite3
from timeit import default_timer as timer



# https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a
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
        tokens = self._stem_tokens(tokens)

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        return tokens


def run():
    print('Downloading NLTK stop words')
    nltk.download('stopwords')

    print('Downloading punkt')
    nltk.download('punkt')

    print('Conneting to SQLite')
    conn = sqlite3.connect("data/raw/database.sqlite")
    df_reviews = pd.read_sql_query("SELECT * FROM Reviews", conn)

    print(f'Dataset contains {df_reviews.shape[0]:n} reviews.')

    df_data = df_reviews[['Text', 'Score']].copy()

    df_data.drop_duplicates(keep='first', inplace=True)
    print(f'Dropped duplicated. Number of reviews: {df_data.shape[0]:n}.')

    preprocessor = TextPreprocessor(n_jobs=4)

    start_time = timer()
    text = preprocessor.transform(df_data[:100000]['Text'])
    end_time = timer()
    duration_secs = end_time - start_time
    print(f'Preprocessing time: {duration_secs:0.2f} secs.')

    # print(text)

    df_data['Sentiment'] = df_data['Score'].map(lambda score: 1 if score > 3 else 0)

    print('Done')


if __name__ == '__main__':
    run()