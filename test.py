# import cPickle as pickle
import pickle

def save_parses(corpus, savepath):
    """
    Parses all sentences in corpus and saves a triple of needed ones in (huge) dictionary
    indexed by sentence number in corpus as pkl object at savepath.

    :corpus: a list of tuples [(chinese sentence, english sentence)] 
    :saves: parse_dict: sentence number -> (target_forest, ref_forest, scr_fsa)   
    """
    parse_dict = dict() 
    for i, (ch_sent, en_sent) in enumerate(corpus):
        lexicon = make_lexicon(ch_sent, ch_en, en_ch)

        src_fsa = libitg.make_fsa(ch_sent)
        src_cfg = libitg.make_source_side_itg(lexicon)
        forest = libitg.earley(src_cfg, src_fsa, 
                               start_symbol=Nonterminal('S'), 
                               sprime_symbol=Nonterminal("D(x)"))

        projected_forest = libitg.make_target_side_itg(forest, lexicon)

        tgt_fsa = libitg.make_fsa(en_sent)
        ref_forest = libitg.earley(projected_forest, tgt_fsa, 
                                   start_symbol=Nonterminal("D(x)"), 
                                   sprime_symbol=Nonterminal('D(x,y)'))

        length_fsa = libitg.LengthConstraint(len(en_sent), strict=False)

        target_forest = libitg.earley(projected_forest, length_fsa, 
                                      start_symbol=Nonterminal("D(x)"), 
                                      sprime_symbol=Nonterminal("D_n(x)"))
        
        parse_dict[i] = (target_forest, ref_forest, src_fsa)
    
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
        lexicon = make_lexicon(ch_sent, ch_en, en_ch)

        src_fsa = libitg.make_fsa(ch_sent)
        src_cfg = libitg.make_source_side_itg(lexicon)
        forest = libitg.earley(src_cfg, src_fsa, 
                               start_symbol=Nonterminal('S'), 
                               sprime_symbol=Nonterminal("D(x)"))

        projected_forest = libitg.make_target_side_itg(forest, lexicon)

        tgt_fsa = libitg.make_fsa(en_sent)
        ref_forest = libitg.earley(projected_forest, tgt_fsa, 
                                   start_symbol=Nonterminal("D(x)"), 
                                   sprime_symbol=Nonterminal('D(x,y)'))

        length_fsa = libitg.LengthConstraint(len(en_sent), strict=False)

        target_forest = libitg.earley(projected_forest, length_fsa, 
                                      start_symbol=Nonterminal("D(x)"), 
                                      sprime_symbol=Nonterminal("D_n(x)"))
        
        parses = (target_forest, ref_forest, src_fsa)
        
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