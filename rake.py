# coding: utf-8

# Implementation of RAKE - Rapid Automtic Keyword Exraction algorithm
# as described in:
# Rose, S., D. Engel, N. Cramer, and W. Cowley (2010).
# Automatic keyword extraction from indi-vidual documents.
# In M. W. Berry and J. Kogan (Eds.), Text Mining: Applications and Theory.unknown: John Wiley and Sons, Ltd.
#
# NOTE: The original code (from https://github.com/aneesha/RAKE)
# has been extended by a_medelyan (zelandiya)
# with a set of heuristics to decide whether a phrase is an acceptable candidate
# as well as the ability to set frequency and phrase length parameters
# important when dealing with longer documents

import itertools
import operator
import re
import string

import nltk
import nltk.data

debug = False
test = False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = nltk.corpus.stopwords.words('english')
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return set(stop_words)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# stop_words = set(nltk.corpus.stopwords.words('english'))
# with open("static/stopwords.lst", "r") as word_list:
#     self.stop_words = self.stop_words.union([w.lower().strip() for w in word_list])


class Memoize:

    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


# extract_candidate_chunks = Memoize(extract_candidate_chunks)


# extract_candidate_words = Memoize(extract_candidate_words)


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    # splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    # for single_word in splitter.split(text):
    for single_word in nltk.word_tokenize(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to
        # invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    global sent_detector
    return sent_detector.sentences_from_text(text)
    # sentence_delimiters = re.compile(
    #     u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    # sentences = sentence_delimiters.split(text)
    # return sentences


def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = '\\b' + word + '\\b'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile(
        '|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


def noop(txt):
    return txt


class Rake(object):

    def __init__(self, stop_words_path, min_char_length=1, max_words_length=5, min_keyword_frequency=1,
                 extractor='chunks', stemming=False):
        self.__stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)
        self.__stop_words = load_stop_words(stop_words_path)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__stem = nltk.stem.lancaster.LancasterStemmer().stem if stemming else noop
        self.__end = re.compile('\W+$', re.UNICODE)
        self.__begin = re.compile('^\W+', re.UNICODE)
        self.__invalid_char = re.compile(u"[^\w\d\-''’`]+",  re.UNICODE)
        self.__extractor = self.__extract_candidate_chunks if extractor == 'chunks' else self.__extract_candidate_words

    def __trim(self, txt):
        txt = self.__invalid_char.sub(' ', txt)
        txt = self.__begin.sub('', txt)
        txt = self.__end.sub('', txt)
        txt = txt.replace('  ', ' ')
        return txt.lower()

    def __generate_candidate_keyword_scores(self, phrase_list, word_score):
        keyword_candidates = {}

        for phrase in phrase_list:
            if self.__min_keyword_frequency > 1:
                if phrase_list.count(phrase) < self.__min_keyword_frequency:
                    continue
            keyword_candidates.setdefault(phrase, 0)
            word_list = separate_words(phrase, self.__min_char_length)
            candidate_score = 0
            for word in word_list:
                try:
                    candidate_score += word_score[word]
                except KeyError:
                    pass
            keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def __calculate_word_scores(self, phraseList):
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = separate_words(phrase, self.__min_char_length)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            # if word_list_degree > 3: word_list_degree = 3 #exp.
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree  # orig.
                # word_degree[word] += 1/(word_list_length*1.0) #exp.
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            # orig.
            if item.lower() in self.__stop_words:
                word_score[item] = 0
            else:
                word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
        # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return word_score

    def __extract_candidate_chunks(self, text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)

        # tokenize, POS-tag, and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(
            sent) for sent in nltk.sent_tokenize(text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
        # join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word, pos, chunk in group).lower()
                      for key, group in itertools.groupby(all_chunks, lambda (word, pos, chunk): chunk != 'O') if key]

        return [self.__trim(cand) for cand in candidates
                if cand not in self.__stop_words
                and not all(char in punct for char in cand)
                and not len(nltk.word_tokenize(cand)) > self.__max_words_length]

    def __extract_candidate_words(self, text, good_tags=set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])):
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        # stop_words = set(nltk.corpus.stopwords.words('english'))
        # tokenize and POS-tag words
        tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                        for sent in nltk.sent_tokenize(text)))
        # filter on certain POS tags and lowercase all words
        candidates = [self.__trim(word) for word, tag in tagged_words
                      if tag in good_tags and word.lower() not in self.__stop_words
                      and not all(char in punct for char in word)]

        return candidates

    def run(self, text):
        sentence_list = split_sentences(text)

        # phrase_list = generate_candidate_keywords(
        # sentence_list, self.__stop_words_pattern, self.__min_char_length,
        # self.__max_words_length)

        phrase_list = self.__extractor(text)

        word_scores = self.__calculate_word_scores(sentence_list)
        # word_scores = calculate_word_scores(sentence_list, self.__min_char_length)

        keyword_candidates = self.__generate_candidate_keyword_scores(phrase_list, word_scores)

        # import ipdb; ipdb.set_trace()

        sorted_keywords = sorted(keyword_candidates.iteritems(
        ), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords

    def summaries(self, text, phrase_list=[], max_word_length=40):
        sentence_list = [s.lower() for s in split_sentences(text)]
        if not phrase_list:
            phrase_list = self.__extractor(text)
        phrase_list = [k.lower() for k in phrase_list]

        word_scores = self.__calculate_word_scores(sentence_list)
        keyword_candidates = self.__generate_candidate_keyword_scores(phrase_list, word_scores)

        scored_sentence_list = []
        for sent in sentence_list:
            score = 0
            went_words = nltk.word_tokenize(sent)
            if len(went_words) > max_word_length:
                continue
            stemmed_sent = ' '.join([self.__stem(w) for w in went_words])
            kw_list = []
            for kw in keyword_candidates:
                # cnt = sent.count(kw)
                cnt = stemmed_sent.count(self.__stem(kw))
                if cnt:
                    score += keyword_candidates[kw] * cnt
                    kw_list.append((kw, keyword_candidates[kw]))
            score /= (len(went_words) / (len(kw_list) or 1))
            scored_sentence_list.append((sent, score))

        return sorted(scored_sentence_list, key=operator.itemgetter(1), reverse=True)


if test:
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."

    # Split text into sentences
    sentenceList = split_sentences(text)
    # stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it
    # will not find "natural numbers" like in Table 1.1
    # SMART stoplist misses some of the lower-scoring keywords in Figure 1.5,
    # which means that the top 1/3 cuts off one of the 4.0 score words in
    # Table 1.1
    stoppath = "RAKE/SmartStoplist.txt"
    stopwordpattern = build_stop_word_regex(stoppath)

    # generate candidate keywords
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)

    # calculate individual word scores
    wordscores = calculate_word_scores(phraseList)

    # generate candidate keyword scores
    keywordcandidates = generate_candidate_keyword_scores(
        phraseList, wordscores)
    if debug:
        print keywordcandidates

    sortedKeywords = sorted(keywordcandidates.iteritems(),
                            key=operator.itemgetter(1), reverse=True)
    if debug:
        print sortedKeywords

    totalKeywords = len(sortedKeywords)
    if debug:
        print totalKeywords
    print sortedKeywords[0:(totalKeywords / 3)]

    rake = Rake("SmartStoplist.txt")
    keywords = rake.run(text)
    print keywords
# [l[i:i+len(pat)]==pat for i in xrange(len(l)-len(pat)+1)].count(True)
