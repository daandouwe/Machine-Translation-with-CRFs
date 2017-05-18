import lib.formal as formal
import lib.libitg as libitg
from lib.formal import Symbol, Terminal, Nonterminal, Span, Rule, CFG, FSA
from lib.earley import earley
from features import featurize_edges
import pickle
from math import nan
import string
from collections import defaultdict
import progressbar

### READ DATA ###

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


def translations(path='data/lexicon', k=5, null=5, remove_punct=True):
    f = open(path, 'r')
    
    ch_en_ = defaultdict(lambda: defaultdict(float))
    en_ch_ = defaultdict(lambda: defaultdict(float))

    for line in f:
        ch, en, p_en_given_ch, p_ch_given_en = line.split()
        # for use in the parsing we replace <NULL> with -EPS-
        if ch == '<NULL>':
            ch = '-EPS-' 
        if en == '<NULL>':
            en = '-EPS-' 
        ch_en_[ch][en] = float(p_en_given_ch) if not p_en_given_ch == 'NA' else 1e-10
        en_ch_[en][ch] = float(p_ch_given_en) if not p_ch_given_en == 'NA' else 1e-10
    f.close()
    
    ch_punct = list("[+\.\!\/_,$%^*(+\"\']+|[+——！，。？? 、~@#￥%……&*（）：；《）《》“”()»〔〕-]+")

    ch_en = defaultdict(lambda: defaultdict(float))
    for ch in ch_en_.keys():
        en_punct = string.punctuation
        srtd = sorted(ch_en_[ch].items(), key=lambda xy: xy[1])
        if ch == '-EPS-':
            if remove_punct:
                # when we do not want to insert punctuation from EPS
                srtd = [(en, p) for en, p in srtd if en not in en_punct]
            ch_en['-EPS-'] = {en: p for en, p in srtd[-null:]}
        else:
            if ch == '。':
                ch_en[ch] = {'.': 1.0} # we always translate a chinese dot into an english dot
            elif ch in ch_punct:
                ch_en[ch] = {ch: 1.0} # we never translate any other punctuation: a fair assumption
            else:
                if remove_punct:
                    # when we do not want to translate a word to punctuation mark
                    srtd = [(en, p) for en, p in srtd if en not in en_punct]

                ch_en[ch] = {en: p for en, p in srtd[-(k-1):]+[('-EPS-', ch_en_[ch]['-EPS-'])]} # each chinese word can be removed
                # NOTE: punctuation can never be removed!


    en_ch = defaultdict(lambda: defaultdict(float))
    for en in en_ch_.keys():
        srtd = sorted(en_ch_[en].items(), key=lambda xy: xy[1])
        if en == '-EPS-=':
            if remove_punct:
                # when we do not want to insert punctuation from EPS
                srtd = [(ch, p) for ch, p in srtd if ch not in ch_punct]
            en_ch['-EPS-'] = {ch: p for ch, p in srtd[-null:]}
        else:
            en_ch[en] = {ch: p for ch, p in srtd[-k:]}
            
    full_en_ch = en_ch_
    full_ch_en = ch_en_
    return ch_en, en_ch, full_en_ch, full_ch_en


def translations_ALT(path='data/lexicon', k=5, null=5, remove_punct=True):

    """
    Other format:
    
    ch_en_[ch][en] = p(ch|en) + p(en|ch)
        or
    ch_en_[ch][en] = p(ch|en) * p(en|ch)
    """

    f = open(path, 'r')
    
    ch_en_ = defaultdict(lambda: defaultdict(float))
    en_ch_ = defaultdict(lambda: defaultdict(float))

    for line in f:
        ch, en, p_en_given_ch, p_ch_given_en = line.split()
        # for use in the parsing we replace <NULL> with -EPS-
        if ch == '<NULL>':
            ch = '-EPS-' 
        if en == '<NULL>':
            en = '-EPS-' 
        if p_en_given_ch == 'NA':
            p_en_given_ch = 1e-10
        if p_ch_given_en == 'NA':
            p_ch_given_en = 1e-10
        ch_en_[ch][en] = float(p_en_given_ch) + float(p_ch_given_en)
    f.close()
    
    ch_punct = list("[+\.\!\/_,$%^*(+\"\']+|[+——！，。？? 、~@#￥%……&*（）：；《）《》“”()»〔〕-]+")

    ch_en = defaultdict(lambda: defaultdict(float))
    for ch in ch_en_.keys():
        en_punct = string.punctuation
        srtd = sorted(ch_en_[ch].items(), key=lambda xy: xy[1])
        if ch == '-EPS-':
            if remove_punct:
                # when we do not want to insert punctuation from EPS
                srtd = [(en, p) for en, p in srtd if en not in en_punct]
            ch_en['-EPS-'] = {en: p for en, p in srtd[-null:]}
        else:
            if ch == '。':
                ch_en[ch] = {'.': 1.0} # we always translate a chinese dot into an english dot
            elif ch in ch_punct:
                ch_en[ch] = {ch: 1.0} # we never translate any other punctuation: a fair assumption
            else:
                if remove_punct:
                    # when we do not want to translate a word to punctuation mark
                    srtd = [(en, p) for en, p in srtd if en not in en_punct]

                ch_en[ch] = {en: p for en, p in srtd[-(k-1):]+[('-EPS-', ch_en_[ch]['-EPS-'])]} # each chinese word can be removed
                # NOTE: punctuation can never be removed!


    en_ch = defaultdict(lambda: defaultdict(float))
    for en in en_ch_.keys():
        srtd = sorted(en_ch_[en].items(), key=lambda xy: xy[1])
        if en == '-EPS-=':
            if remove_punct:
                # when we do not want to insert punctuation from EPS
                srtd = [(ch, p) for ch, p in srtd if ch not in ch_punct]
            en_ch['-EPS-'] = {ch: p for ch, p in srtd[-null:]}
        else:
            en_ch[en] = {ch: p for ch, p in srtd[-k:]}
            
    full_en_ch = en_ch_
    full_ch_en = ch_en_
    return ch_en, en_ch, full_en_ch, full_ch_en



### LEXICON ###

def make_lexicon(ch_sent, ch_en):
    """
    Given a chinese sentence produces a lexicon of possible translation as dictionary
    Format: chinese character -> {top 5 english translations}
    :param ch_sent: a chinese sentence as string (e.g. '在 门厅 下面 。 ')
    """
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(ch_en['-EPS-'])
    for char in ch_sent.split():
        lexicon[char].update(ch_en[char])
    return lexicon

def make_lexicon_ALT(ch_sent, en_sent):
    """
    Given a chinese sentence produces a lexicon of possible translation as dictionary
    :param ch_sent: a chinese sentence as string (e.g. '在 门厅 下面 。 ')
    :param en_sent: a english sentence as string (e.g. 'it 's just down the hall .')
    Format: chinese character -> {all english words in en_sent}
    """
    lexicon = defaultdict(set)
    lexicon['-EPS-'].update(en_sent.split())
    for char in ch_sent.split():
        lexicon[char].update(en_sent.split())
        lexicon[char].add('-EPS-')
    return lexicon


def make_total_lexicon(lexicons):
    """
    Takes a list of lexicons and return one large lexion holding all key-value pairs
    for each lexicon in lexicons.
    :returns: a lexicon in the format word -> {translations}
    """
    full_lexicon = defaultdict(set)
    for lexicon in lexicons:
        for src, translations in lexicon.items():
            full_lexicon[src].update(translations)
    return full_lexicon


### PARSING ###

def parse_forests(src_sent, tgt_sent, lexicon):
    """
    Parses src_sent and tgt_sent and returns all the forests needed for sgd.
    Note: uses the length constraint approach.
    """
    src_fsa = libitg.make_fsa(src_sent)
    src_cfg = libitg.make_source_side_itg(lexicon)
    forest = earley(src_cfg, src_fsa, 
                    start_symbol=Nonterminal('S'), 
                    sprime_symbol=Nonterminal("D(x)"),
                    clean=True)
    
    projected_forest = libitg.make_target_side_itg(forest, lexicon)
    
    tgt_fsa = libitg.make_fsa(tgt_sent)
    ref_forest = earley(projected_forest, tgt_fsa, 
                        start_symbol=Nonterminal("D(x)"), 
                        sprime_symbol=Nonterminal('D(x,y)'),
                        eps_symbol=Nonterminal('-EPS-'))
    
    length_fsa = libitg.LengthConstraint(len(src_sent.split()), strict=False)
    target_forest = earley(projected_forest, length_fsa, 
                           start_symbol=Nonterminal("D(x)"), 
                           sprime_symbol=Nonterminal("D_n(x)"))
    
    return target_forest, ref_forest, src_fsa

def parse_forests_eps(src_sent, tgt_sent, lexicon):
    """
    Parses src_sent and tgt_sent and returns all the forests needed for sgd.
    Note: uses the alternative epsilon-insertion constraint.
    """
    src_fsa = libitg.make_fsa(src_sent)
    src_cfg = libitg.make_source_side_itg(lexicon)
    _Dx = earley(src_cfg, src_fsa, 
                 start_symbol=Nonterminal('S'), 
                 sprime_symbol=Nonterminal("D(x)"),
                 clean=True)
    
    eps_count_fsa = libitg.InsertionConstraint(3)

    _Dix = earley(_Dx, eps_count_fsa, 
                  start_symbol=Nonterminal('D(x)'), 
                  sprime_symbol=Nonterminal('D_n(x)'), 
                  eps_symbol=None)

    target_forest = libitg.make_target_side_itg(_Dix, lexicon)

    tgt_fsa = libitg.make_fsa(tgt_sent)

    ref_forest = earley(target_forest, tgt_fsa, 
                        start_symbol=Nonterminal("D_n(x)"), 
                        sprime_symbol=Nonterminal('D(x,y)'))
    
    return target_forest, ref_forest, src_fsa, tgt_sent


### SAVING AND LOADING ###

def save_parses(corpus, lexicon, savepath):
    """
    Parses all sentences in corpus and saves a triple of needed ones in (huge) dictionary
    indexed by sentence number in corpus as pkl object at savepath.

    :corpus: a list of tuples [(chinese sentence, english sentence)] 
    :param lexicon: a lexicon holding translations for each word in the corpus
    :saves: parse_dict: sentence number -> (target_forest, ref_forest, scr_fsa)   
    """
    parse_dict = dict() 
    for i, (ch_sent, en_sent) in enumerate(corpus):
        parses = parse_forests(ch_sent, en_sent, lexicon)        
        parse_dict[i] = parses 
    f = open(savepath + 'parse-dict.pkl', 'wb')
    pickle.dump(parse_dict, f, protocol=4)
    f.close()


def save_parses_separate(corpus, lexicon, savepath, src_tgt, tgt_src, eps=True, sparse=True):
    """
    For each sentence k in corpus we parse and save the triple of needed parses 
    as pkl object at savepath/parses-k.pkl.

    :corpus: a list of tuples [(chinese sentence, english sentence)] 
    :param lexicon: a lexicon holding translations for each word in the corpus
    :saves: parses-k = (target_forest, ref_forest, scr_fsa, tgt_sent) for each k in 0,..,len(corpus)
    :returns fset: all features used in both the forests
    """
    fset = set()
    print('Parsing...')
    bar = progressbar.ProgressBar(max_value=len(corpus))

    for k, (src_sent, tgt_sent) in enumerate(corpus):
        bar.update(k)
        if eps:
            parses = parse_forests_eps(src_sent, tgt_sent, lexicon)
        else:
            parses = parse_forests(src_sent, tgt_sent, lexicon)
        f = open(savepath + 'parses-{}.pkl'.format(k), 'wb')
        pickle.dump(parses, f, protocol=4)
        f.close()

        # update fset
        tgt_forest, ref_forest, src_fsa, tgt_sent = parses
        _, fset1 = featurize_edges(ref_forest, src_fsa, 
                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse, src_tgt=src_tgt, tgt_src=tgt_src)
        _, fset2 = featurize_edges(tgt_forest, src_fsa, 
                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse, src_tgt=src_tgt, tgt_src=tgt_src)
        fset.update(fset1 | fset2)
        
        bar.update(k+1)

    bar.finish()
    return fset


def load_parses(savepath):
    """
    Loads and returns a parse_dict as saved by load_parses.
    """
    f = open(savepath + 'parse-dict.pkl', 'rb')
    parse_dict = pickle.load(f)
    f.close()
    return parse_dict


def load_parses_separate(savepath, k):
    """
    Loads and returns parses as saved by save_parses_separate
    """
    f = open(savepath + 'parses-{}.pkl'.format(k), 'rb')
    parse = pickle.load(f)
    f.close()
    return parse


def save_lexicon(lexicon, savepath):
    f = open(savepath + 'lexicon.pkl', 'wb')
    pickle.dump(lexicon, f, protocol=4)
    f.close()


def load_lexicon(savepath):
    f = open(savepath + 'lexicon.pkl', 'rb')
    lexicon = pickle.load(f)
    f.close()
    return lexicon


def save_featureset(fset, savepath):
    f = open(savepath + 'feature-set.pkl', 'wb')
    pickle.dump(fset, f, protocol=4)
    f.close()


def load_featureset(savepath):
    f = open(savepath + 'feature-set.pkl', 'rb')
    fset = pickle.load(f)
    f.close()
    return fset


