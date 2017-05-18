import lib.formal
from lib.formal import Symbol, Terminal, Nonterminal, Span, Rule, CFG, FSA
from graph import *
from collections import defaultdict
import numpy as np


# def get_terminal_string(symbol: Symbol):
#     """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
#     if not symbol.is_terminal():
#         raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
#     return symbol.root().obj()


# def get_bispans(symbol: Span):
#     """
#     Returns the bispans associated with a symbol. 
    
#     The first span returned corresponds to paths in the source FSA (typically a span in the source sentence),
#      the second span returned corresponds to either
#         a) paths in the target FSA (typically a span in the target sentence)
#         or b) paths in the length FSA
#     depending on the forest where this symbol comes from.
#     """
#     if not isinstance(symbol, Span):
#         raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
#     s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
#     _, start1, end1 = s.obj()  # this unwraps the source annotation
#     return (start1, end1), (start2, end2)


def get_source_word(fsa: FSA, origin: int, destination: int) -> str:
    """Returns the python string representing a source word from origin to destination (assuming there's a single one)"""
    labels = list(fsa.labels(origin, destination))
    if len(labels) == 0:
        return '-EPS-'
    assert len(labels) == 1, 'Use this function only when you know the path is unambiguous, found %d labels %s for (%d, %d)' % (len(labels), labels, origin, destination)
    return labels[0]

def get_target_word(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
    return symbol.root().obj()

def get_bispans(symbol: Span):
    """
    Returns the bispans associated with a symbol. 
    
    The first span returned corresponds to paths in the source FSA (typically a span in the source sentence),
     the second span returned corresponds to either
        a) paths in the target FSA (typically a span in the target sentence)
        or b) paths in the length FSA
    depending on the forest where this symbol comes from.
    """
    if not isinstance(symbol, Span):
        raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
    s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    s1, start1, end1 = s.obj()  # this unwraps the source annotation
    if isinstance(s1, Span): # for the (weird) case of triple spans on symbols: '-EPS-':3-4:0-0:4-4 and [S]:0-2:0-0:0-2 for example.
        _, start1, end1 = s1.obj()
        # print(symbol)
        # print(s1)
        # print(type(s1))
        # print(start1, end1)
    return (start1, end1), (start2, end2)


def simple_features(edge: Rule, src_fsa: FSA, eps=Terminal('-EPS-'), 
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                   src_tgt=defaultdict(lambda:defaultdict(float)), 
                   tgt_src=defaultdict(lambda:defaultdict(float))) -> dict:
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    crucially, note that the target sentence y is not available!    
    """
    fmap = defaultdict(float)
    fset = set() # stores the features we've added
    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        fset.add('type:binary')
        # here we could have sparse features of the source string as a function of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS        
        
        # TODO: double check these, assign features, add some more
        if ls1 == ls2:  # deletion of source left child
            fmap['type:deletion-slc'] += 1.0
            fset.add('type:deletion-slc')
        if rs1 == rs2:  # deletion of source right child
            fmap['type:deletion-src'] += 1.0
            fset.add('type:deletion-src')
        if ls2 == rs1:  # monotone
            fmap['type:monotone'] += 1.0
            fset.add('type:monotone')
        if ls1 == rs2:  # inverted
            fmap['type:inverted'] += 1.0
            fset.add('type:inverted')
            
        # add features: 
        #    type: inverted:span
        #    type: monotone:span

        # source span feature of rhs
        src_span_lc = ls2 - ls1
        src_span_rc = rs2 - rs1
        fmap['span:rhs:src-lc:{}'.format(src_span_lc)] += 1.0
        fmap['span:rhs:src-rc:{}'.format(src_span_rc)] += 1.0
        fset.update({'span:rhs:src-lc:{}'.format(src_span_lc),
                  'span:rhs:src-rc:{}'.format(src_span_rc)})
        # target span feature of rhs
        tgt_span_lc = lt2 - lt1
        tgt_span_rc = rt2 - rt1
        fmap['span:rhs:tgt-lc:{}'.format(tgt_span_lc)] += 1.0
        fmap['span:rhs:tgt-rc:{}'.format(tgt_span_rc)] += 1.0
        fset.update({'span:rhs:tgt-lc:{}'.format(tgt_span_lc),
                  'span:rhs:tgt-rc:{}'.format(tgt_span_rc)})
        
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            fset.add('type:terminal')
            # we could have IBM1 log probs for the traslation pair or ins/del
            (s1, s2), (t1, t2) = get_bispans(symbol)
            # src_word = get_source_word(src_fsa, s1, s2)
            # tgt_word = get_terminal_string(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                # for sure there is a source word
                src_word = get_source_word(src_fsa, s1, s2)
                fmap['type:deletion'] += 1.0
                fset.add('type:deletion')
                # dense versions (for initial development phase)
                # fmap['ibm1:del:logprob'] += tgt_src[src_word]['<NULL>']
                
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
                    fset.add('del:%s' % src_word)
            else:
                # for sure there's a target word
                tgt_word = get_target_word(symbol)                
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    fset.add('type:insertion')
                    # dense version
                    # fmap['ibm1:ins:logprob'] += src_tgt['<NULL>'][tgt_word]
                    
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                        fset.add('ins:%s' % tgt_word)
                else:
                    # for sure there's a source word
                    src_word = get_source_word(src_fsa, s1, s2)
                    fmap['type:translation'] += 1.0
                    fset.add('type:translation')
                    # dense version
                    # fmap['ibm1:x2y:logprob'] += src_tgt[src_word][tgt_word]  # y is english word 
                    # fmap['ibm1:y2x:logprob'] += tgt_src[tgt_word][src_word]
                    
                    # sparse version                    
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
                        fset.add('trans:%s/%s' % (src_word, tgt_word))
        
                    # add features for source skip-bigram
                    l_word = '-START-' if s1 == 0 else get_source_word(src_fsa, s1-1, s1)
                    r_word = '-END-' if s2+1 == src_fsa.nb_states() else get_source_word(src_fsa, s2, s2+1)
                    skip_feature = 'skip-bigram:{0}*{1}'.format(l_word, r_word)
                    fmap[skip_feature] += 1
                    fset.add(skip_feature)

            # source span feature of rhs
            src_span = s2 - s1
            fmap['span:rhs:src:{}'.format(src_span)] += 1.0
            fset.add('span:rhs:src:{}'.format(src_span))
            # target span feature of rhs
            tgt_span = t2 - t1
            fmap['span:rhs:tgt:{}'.format(tgt_span)] += 1.0
            fset.add('span:rhs:tgt:{}'.format(tgt_span))
                
        else:  # S -> X
            fmap['top'] += 1.0
            fset.add('top')

        # bispans of lhs of edge for source and target (source and target sentence lengths)
        if isinstance(edge.lhs.obj()[0], Span): # exclude the (Nonterminal('D(x)'), 0, 2) rules
            (s1, s2), (t1, t2) = get_bispans(edge.lhs)
            # source span feature of lhs
            src_span = s2 - s1
            fmap['span:lhs:src:{}'.format(src_span)] += 1.0
            fset.add('span:lhs:src:{}'.format(src_span))
            # target span feature of lhs
            tgt_span = t2 - t1
            fmap['span:lhs:tgt:{}'.format(tgt_span)] += 1.0
            fset.add('span:lhs:tgt:{}'.format(tgt_span))

    return fmap, fset


def get_full_fset(parses, src_tgt, tgt_src, sparse=False):
    """
    Returns the full feature set of target_forest and ref_forest together in one set

    :param parses: a set of parses format [(tgt_forest, ref_forest, src_fsa)]
    :param src_tgt: a dictionary with translation probabilities src_tgt[scr_word][tgt_word] = p(tgt_word|src_word)
    :param tgt_src: a dictionary with translation probabilities tgt_tgt[tgt_word][src_word] = p(src_word|tgt_word)
    :returns fset: a the set of all features used in all of the forests in `parses` together in one set
    """
    fset = set()
    for tgt_forest, ref_forest, src_fsa in parses:
        _, fset2 = featurize_edges(ref_forest, src_fsa, sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse, src_tgt=src_tgt, tgt_src=tgt_src)
        _, fset1 = featurize_edges(tgt_forest, src_fsa, sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse, src_tgt=src_tgt, tgt_src=tgt_src)
        fset.update(fset1 | fset2)
    return fset


def featurize_edges(forest, src_fsa, 
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                    src_tgt=defaultdict(lambda:defaultdict(float)), 
                    tgt_src=defaultdict(lambda:defaultdict(float)),
                    eps=Terminal('-EPS-')) -> dict:
    edge2fmap = defaultdict()
    fset_accum = set()
    for edge in forest:
        edge2fmap[edge], fset = simple_features(edge, src_fsa, eps, sparse_del, sparse_ins, sparse_trans)
        fset_accum.update(fset)
    return edge2fmap, fset_accum


def weight_function(edge, fmap, wmap) -> float:
    # dot product of fmap and wmap  (working in log-domain)
    dot = 0.0
    for feature, value in fmap.items():
        dot += value * wmap[feature]
    return dot


def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features: dict) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    expected_features = defaultdict(lambda:defaultdict(float))
    for rule in forest:
        k = outside[rule.lhs]
        for symbol in rule.rhs:
            k *= inside[symbol]
        for feature in edge_features[rule]:
            expected_features[rule][feature] = k * edge_features[rule][feature]
    return expected_features


def expected_feature_vector_log(forest: CFG, inside: dict, outside: dict, edge_features: dict) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    expected_features = defaultdict(lambda:defaultdict(float))
    for rule in forest:
        k = outside[rule.lhs]
        for symbol in rule.rhs:
            k += inside[symbol]
        for feature in edge_features[rule]:
            expected_features[rule][feature] = k + edge_features[rule][feature]
    return expected_features


