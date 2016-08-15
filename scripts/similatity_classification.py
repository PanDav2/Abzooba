import os
import utils as ut
import logging
import pandas as pd

import numpy as np
# ML Libraries
from sklearn.feature_extraction import text as txt
import sklearn.metrics.pairwise as metrics

## NLP Libraries
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class SimClass(object):
    """
    This class allows the user to retrieve a ranked list of document for a given, containing the most similar
    """

    def __init__(self, corpus_dir_path, number_of_similar=5):
        logging.info('loading corpus; please make sure all data is in this corpus')
        self.raw_data = list()
        for _, _, files in os.walk(corpus_dir_path):
            self.raw_data = [read_file(corpus_dir_path + '/' + f) for f in files]
        self.df = pd.DataFrame(self.raw_data, columns=['text'])
        logging.info("checking file_encoding")
        ## Checking the stopwords value :
        self.stop_words_ = get_nltk_stop_words(['english'])
        self.number_of_similar = number_of_similar

    def fit(self, strip_accents=u'unicode', lowercase=True,
            preprocessor=None, stop_words=None, token_pattern=u"[\\w']+\\w\\b",
            analyzer=u'word', max_df=1.0, max_features=20000, vocabulary=None,
            binary=False, ngram_range=(1, 1), min_df=1,
            normalize=True, decode_error='ignore'):
        """ From a document list, create a matrice containinga a sparse representation of
        Thanks to the hashing table, we obtain an ID associated to each word
        :rtype: object
        """

        assert self.df['text'] is not None, logging("Couldn't find loaded data, please give a valid directory")
        logging.info('fitting a corpus of {} documents in memory'.format(len(self.df['text'])))
        if stop_words is None:
            stop_words = self.stop_words_
        self.vec_params_ = txt.CountVectorizer(strip_accents=strip_accents, lowercase=lowercase,
                                               preprocessor=preprocessor,
                                               stop_words=stop_words, token_pattern=token_pattern, analyzer=analyzer,
                                               max_df=max_df,
                                               max_features=max_features, vocabulary=vocabulary, binary=binary,
                                               ngram_range=ngram_range,
                                               min_df=min_df, decode_error=decode_error)
        self.bow_ = self.vec_params_.fit_transform(self.df['text'])
        self.bow_ = self.bow_.tocsr()  # permet de print
        self.normalize = normalize
        if self.normalize:
            self.bow_ = self.normalize_bow(self.bow_)

    def normalize_bow(self, bow=None):
        logging.info('using a normalized representation')
        if bow is None:
            logging.info('no bow given, using the object bow')
            bow = self.bow
        transformer = txt.TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
        bow = transformer.fit_transform(bow)
        return bow

    def transform(self, document):
        document_bow = self.vec_params_.transform([document])
        if self.normalize:
            return self.normalize_bow(document_bow)
        else:
            return document_bow

    def predict(self, document):
        try:
            logging.info('predicting similatiry of text document using cosine similarity')
            # document = treat_item(document)
            fitted_document_ = self.transform(document)
            return map(lambda x: metrics.cosine_similarity(fitted_document_, x), self.bow_)
        except TypeError as e:
            logging.error(e)

    def rank_items(self, results, pos=True):
        results = map(lambda x: x[0][0], results)
        assert isinstance(results, list), logging.error("the given argument to rank item isn't a list")
        results = np.array(results)
        if pos:
            ranked_results_idx_ = np.argsort(results, axis=0, )[::-1][:self.number_of_similar]
            res_df = pd.DataFrame()
            res_df['text'] = [self.df['text'][i] for i in ranked_results_idx_]
            res_df['score'] = results[ranked_results_idx_]
            return res_df
        else:
            ranked_results_idx_ = np.argsort(results, axis=0, )[:self.number_of_similar]
            res_df = pd.DataFrame()
            res_df['text'] = [self.df['text'][i] for i in ranked_results_idx_]
            res_df['score'] = results[ranked_results_idx_]
            return res_df


def treat_item(document):
    if isinstance(document, basestring):
        return [document]
    elif isinstance(document, pd.DataFrame):
        return document['text']
    elif isinstance(document, list):
        return document
    else:
        logging.error("the given document type is not handled")
        raise TypeError("the given document type is not handled")


def get_nltk_stop_words(languages=['english']):
    stop_words = list()
    for l in languages:
        [stop_words.append(w.encode('utf-8')) for w in stopwords.words(l)]
    return stop_words


def read_file(filepath):
    with open(filepath, 'r') as inp:
        return ''.join(inp.readlines())


def main():
    ut.init_logging()
    sc = SimClass('/Users/david/Abzooba/data', 5)
    sc.fit()
    a = sc.predict(sc.raw_data[3])
    logging.info("Most similar texts !\n\n")
    res = sc.rank_items(a)
    logging.info(res['text'].iloc[0])
    logging.info(res['text'].iloc[1])
    logging.info(res['text'].iloc[2])
    logging.info(res['text'].iloc[3])
    logging.info(res['text'].iloc[4])

    logging.info("Less similar texts !\n\n")
    res = sc.rank_items(a, pos=False)
    logging.info(res['text'].iloc[0])
    logging.info(res['text'].iloc[1])
    logging.info(res['text'].iloc[2])
    logging.info(res['text'].iloc[3])
    logging.info(res['text'].iloc[4])


if __name__ == '__main__':
    main()
