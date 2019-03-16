import os
import time
import re
import html
from collections import Counter

import nltk
import numpy as np


def preprocess_sentence(text, keep_most=False):
    """
    Helper function to remove html, unneccessary spaces and punctuation.
    Args:
        text: String.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.

    Returns:
        processed text.

    """
    text = text.lower()
    text = fixup(text)
    text = re.sub(r"<br />", " ", text)
    if keep_most:
        text = re.sub(r"[^a-z0-9%!?.,:()/]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"    ", " ", text)
    text = re.sub(r"   ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = text.strip()
    return text


def fixup(x):
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def preprocess(text, keep_most=False):
    """
    Splits the text into sentences, preprocesses
       and tokenizes each sentence.
    Args:
        text: String. multiple sentences.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.

    Returns:
        preprocessed and tokenized text.

    """
    tokenized = []
    for sentence in nltk.sent_tokenize(text):
        sentence = preprocess_sentence(sentence, keep_most)
        sentence = nltk.word_tokenize(sentence)
        for token in sentence:
            tokenized.append(token)

    return tokenized


def preprocess_texts_and_summaries(texts,
                                   summaries,
                                   keep_most=False):
    """iterates given list of texts and given list of summaries and tokenizes every
       review using the tokenize_review() function.
       apart from that we count up all the words in the texts and summaries.
       returns: - processed texts
                - processed summaries
                - array containing all the unique words together with their counts
                  sorted by counts.
    """

    start_time = time.time()
    processed_texts = []
    processed_summaries = []
    words = []

    for text in texts:
        text = preprocess(text, keep_most)
        for word in text:
            words.append(word)
        processed_texts.append(text)
    for summary in summaries:
        summary = preprocess(summary, keep_most)
        for word in summary:
            words.append(word)

        processed_summaries.append(summary)
    words_counted = Counter(words).most_common()
    print('Processing Time: ', time.time() - start_time)

    return processed_texts, processed_summaries, words_counted


def create_word_inds_dicts(words_counted,
                           specials=None,
                           min_occurences=0):
    """ creates lookup dicts from word to index and back.
        returns the lookup dicts and an array of words that were not used,
        due to rare occurence.
    """
    missing_words = []
    word2ind = {}
    ind2word = {}
    i = 0

    if specials is not None:
        for sp in specials:
            word2ind[sp] = i
            ind2word[i] = sp
            i += 1

    for (word, count) in words_counted:
        if count >= min_occurences:
            word2ind[word] = i
            ind2word[i] = word
            i += 1
        else:
            missing_words.append(word)

    return word2ind, ind2word, missing_words


def convert_sentence(review, word2ind):
    """ converts the given sent to int values corresponding to the given word2ind"""
    inds = []
    unknown_words = []

    for word in review:
        if word in word2ind.keys():
            inds.append(int(word2ind[word]))
        else:
            inds.append(int(word2ind['<UNK>']))
            unknown_words.append(word)

    return inds, unknown_words


def convert_to_inds(input, word2ind, eos=False, sos=False):
    converted_input = []
    all_unknown_words = set()

    for inp in input:
        converted_inp, unknown_words = convert_sentence(inp, word2ind)
        if eos:
            converted_inp.append(word2ind['<EOS>'])
        if sos:
            converted_inp.insert(0, word2ind['<SOS>'])
        converted_input.append(converted_inp)
        all_unknown_words.update(unknown_words)

    return converted_input, all_unknown_words


def convert_inds_to_text(inds, ind2word, preprocess=False):
    """ convert the given indexes back to text """
    words = [ind2word[word] for word in inds]
    return words


def load_pretrained_embeddings(path):
    """loads pretrained embeddings. stores each embedding in a
       dictionary with its corresponding word
    """
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding_vector = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding_vector
    return embeddings


def create_and_save_embedding_matrix(word2ind,
                                     pretrained_embeddings_path,
                                     save_path,
                                     embedding_dim=300):
    """creates embedding matrix for each word in word2ind. if that words is in
       pretrained_embeddings, that vector is used. otherwise initialized
       randomly.
    """
    pretrained_embeddings = load_pretrained_embeddings(pretrained_embeddings_path)
    embedding_matrix = np.zeros((len(word2ind), embedding_dim), dtype=np.float32)
    for word, i in word2ind.items():
        if word in pretrained_embeddings.keys():
            embedding_matrix[i] = pretrained_embeddings[word]
        else:
            embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embedding_matrix[i] = embedding
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, embedding_matrix)
    return np.array(embedding_matrix)
