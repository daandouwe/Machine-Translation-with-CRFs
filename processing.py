import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from collections import defaultdict
import numpy as np
from math import nan
import string

def preprocess_lexicon(path='data/lexicon', k=5, null=5, remove_punct=True):
    f = open(path, 'r')

    
    ch_en_ = defaultdict(lambda: defaultdict())
    en_ch_ = defaultdict(lambda: defaultdict())

    for line in f:
        ch, en, p_en_given_ch, p_ch_given_en = line.split()
        # for use in the parsing we replace <NULL> with -EPS-
        if ch == '<NULL>':
            ch = '-EPS-' 
        if en == '<NULL>':
            en = '-EPS-' 
        ch_en_[ch][en] = float(p_en_given_ch) if not p_en_given_ch == 'NA' else nan # perhaps something tiny like 10e-20
        en_ch_[en][ch] = float(p_ch_given_en) if not p_ch_given_en == 'NA' else nan # perhaps something tiny like 10e-20
    ch_en = defaultdict(lambda: defaultdict())
    
    ch_en = defaultdict(lambda: defaultdict())
    for ch in ch_en_.keys():
        en_punct = string.punctuation
        srtd = sorted(ch_en_[ch].items(), key=lambda xy: xy[1])
        if ch == '-EPS-':
            if remove_punct: 
                srtd = [(en, p) for en, p in srtd if en not in en_punct]
            ch_en['-EPS-'] = {en: p for en, p in srtd[-null:]}
        else:
            ch_en[ch] = {en: p for en, p in srtd[-k:]}


    en_ch = defaultdict(lambda: defaultdict())
    for en in en_ch_.keys():
        ch_punct = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？? 、~@#￥%……&*（）：；《）《》“”()»〔〕-]+"
        srtd = sorted(en_ch_[en].items(), key=lambda xy: xy[1])
        if en == '-EPS-=':
            if remove_punct: 
                srtd = [(ch, p) for ch, p in srtd if ch not in ch_punct]
            en_ch['-EPS-'] = {ch: p for ch, p in srtd[-null:]}
        else:
            en_ch[en] = {ch: p for ch, p in srtd[-k:]}
            
    full_en_ch = en_ch_
    full_ch_en = ch_en_
    return ch_en, en_ch, full_en_ch, full_ch_en


def read_data_dev(path='data/dev1.zh-en', max_sents=5):
    f = open(path, 'r')
    corpus = dict()
    for k, line in enumerate(f):
        if k + 1 > max_sents:
            break
        sents = line[:-1].split('|||')
        ch = sents[0]
        translations = list()
        for en in sents[1:]:
#             translations.append(en.split())
            translations.append(en)
        corpus[ch] = translations
    return corpus

def read_data(path='data/training.zh-en', max_sents=5):
    f = open(path, 'r')
    corpus = list()
    for k, line in enumerate(f):
        if k + 1 > max_sents:
            break
        ch, en = line[:-1].split('|||')
        corpus.append((ch, en))
    return corpus

def make_lexicon(ch_sent, ch_en, en_ch):
    """
    Given a chinese sentence produces a lexicon of possible translation as dictionary
    Format: chinese character -> {top 5 english translations}
    :param ch_sent: a chinese sentence as string (e.g. '一 跳 一 跳 的 痛 。 ')
    """
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(ch_en['-EPS-'])
    for char in ch_sent.split():
        lexicon[char].update(ch_en[char])
    return lexicon

def make_lexicon_ALT(ch_sent, en_sent):
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(en_sent.split())
    for char in ch_sent.split():
        lexicon[char].update(en_sent.split())
    return lexicon

