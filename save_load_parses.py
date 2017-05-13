import pickle
import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from collections import defaultdict
import numpy as np
from features import *
from processing import *
from inside_outside import *

# import cPickle as pickle
import pickle

def parse_forests(src_sent, tgt_sent, lexicon):
    """
    Parses the forests needed for epoch
    """
    src_fsa = libitg.make_fsa(src_sent)
    src_cfg = libitg.make_source_side_itg(lexicon)
    forest = libitg.earley(src_cfg, src_fsa, 
                           start_symbol=Nonterminal('S'), 
                           sprime_symbol=Nonterminal("D(x)"))
    
    projected_forest = libitg.make_target_side_itg(forest, lexicon)
    
    tgt_fsa = libitg.make_fsa(tgt_sent)
    ref_forest = libitg.earley(projected_forest, tgt_fsa, 
                               start_symbol=Nonterminal("D(x)"), 
                               sprime_symbol=Nonterminal('D(x,y)'),
                               eps_symbol=Nonterminal('-EPS-'))
    
    length_fsa = libitg.LengthConstraint(len(src_sent.split()), strict=False)
    target_forest = libitg.earley(projected_forest, length_fsa, 
                                  start_symbol=Nonterminal("D(x)"), 
                                  sprime_symbol=Nonterminal("D_n(x)"))
    
    return target_forest, ref_forest, src_fsa


def save_parses(corpus, savepath):
    """
    Parses all sentences in corpus and saves a triple of needed ones in (huge) dictionary
    indexed by sentence number in corpus as pkl object at savepath.

    :corpus: a list of tuples [(chinese sentence, english sentence)] 
    :saves: parse_dict: sentence number -> (target_forest, ref_forest, scr_fsa)   
    """
    parse_dict = dict() 
    for i, (ch_sent, en_sent) in enumerate(corpus):
        lexicon = make_lexicon_ALT(ch_sent, en_sent)

        parses = parse_forests(ch_sent, en_sent, lexicon)
        
        parse_dict[i] = parses
    
    f = open(savepath + 'parse_dict.pkl', 'wb')
    pickle.dump(parse_dict, f, protocol=4)
    f.close()

def save_parses_separate(corpus, savepath):
    """
    For each sentence k in corpus we parse and save the triple of needed parses 
    as pkl object at savepath/parses-k.pkl.

    :corpus: a list of tuples [(chinese sentence, english sentence)] 
    :saves: parses-k = (target_forest, ref_forest, scr_fsa) for each k in 0,..,len(corpus)
    """
    for k, (ch_sent, en_sent) in enumerate(corpus):
        lexicon = make_lexicon_ALT(ch_sent, en_sent)

        parses = parse_forests(ch_sent, en_sent, lexicon)
        
        f = open(savepath + 'parses-{}.pkl'.format(k), 'wb')
        pickle.dump(parses, f, protocol=4)
        f.close()
    
def load_parses(savepath):
    """
    Loads and returns a parse_dict as saved by load_parses.
    """
    f = open(savepath + 'parse_dict.pkl', 'rb')
    parse_dict = pickle.load(f)
    f.close()
    return parse_dict

def load_parses_separate(savepath, k):
    """
    Loads and returns parses as saved by load_parses_separate
    """
    f = open(savepath + 'parses-{k}.pkl', 'rb')
    parses = pickle.load(f)
    f.close()
    return parses

corpus = read_data(max_sents=100)
save_parses_separate(corpus[:100], '../parses/')
