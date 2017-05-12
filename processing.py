from collections import defaultdict

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
            translations.append(en)
        corpus[ch] = translations
    return corpus

def read_data(path='data/training.zh-en', max_sents=5):
    f = open(path, 'r')
    corpus = list()
    for k, line in enumerate(f):
        if k + 1 > max_sents:
            break
        ch, en = line[:-1].split('|||') # line[:-1] to remove '\n' character
        corpus.append((ch, en))
    return corpus

def make_lexicon(ch_sent, ch_en, en_ch):
    """
    Given a chinese sentence produces a lexicon of possible translation as dictionary
    Format: chinese character -> {top 5 english translations}
    :param ch_sent: a chinese sentence as string (e.g. '在 门厅 下面 。 ')
    :param ch_sent: a chinese sentence as string (e.g. 'it 's just down the hall .')
    """
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(ch_en['-EPS-'])
    for char in ch_sent.split():
        lexicon[char].update(ch_en[char])
    return lexicon

def make_lexicon_ALT(ch_sent, en_sent):
    """
    Given a chinese sentence produces a lexicon of possible translation as dictionary
    Format: chinese character -> {all english words in training sentence}
    """
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(en_sent.split())
    for char in ch_sent.split():
        lexicon[char].update(en_sent.split())
        lexicon[char].add('-EPS-')
    return lexicon