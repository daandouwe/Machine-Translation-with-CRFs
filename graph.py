import lib.formal
from lib.formal import Symbol, Terminal, Nonterminal, Span, Rule, CFG, FSA
from util import write_derrivation
from collections import defaultdict, deque, Counter
import numpy as np


def top_sort(forest: CFG) -> list:
    """Returns ordered list of nodes according to topsort order in an acyclic forest"""
    S = {symbol for symbol in forest.terminals} # (Copy!) only terminals have no dependecies
    D = {symbol: {child for rule in forest.get(symbol) for child in rule.rhs}\
                 for symbol in forest.nonterminals|forest.terminals} # forest.nonterminals|forest.terminals = V
    L = list()
    while S: # while S nonempty
        u = S.pop()
        L.append(u)
        outgoing = [e for e in forest if u in e.rhs] # outgoing = FS(u)
        for rule in outgoing:
            v = rule.lhs
            D[v] = D[v] - {u}
            if len(D[v]) == 0:
                S = S | {v}
    return L

def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node"""
    I = dict()
    for symbol in tsort: # symbol is v
        incoming = forest.get(symbol) # BS(v) - gets all the incoming nodes, i.e. all rules where symbol is lhs
        if len(incoming) == 0: 
            I[symbol] = 1.0 # leaves
        else:
            w = 0.0
            for edge in incoming: # edge is of type Rule
                k = edge_weights[edge]
                for child in edge.rhs: # chid in tail(e)
                    k *= I[child] # TODO: change to log-sum-exp
                w += k
            I[symbol] = w
    return I


def inside_algorithm_log(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node"""
    I = dict()
    for symbol in tsort: # symbol is v
        incoming = forest.get(symbol) # BS(v) - gets all the incoming nodes, i.e. all rules where symbol is lhs
        if len(incoming) == 0: 
            I[symbol] = 0.0 # leaves
        else:
            # w = 0.0
            w = -np.inf
            parts = []
            for edge in incoming: # edge is of type Rule
                k = edge_weights[edge]
                for child in edge.rhs: # chid in tail(e)
                    k += I[child]
                # w = np.log(np.exp(w) + np.exp(k))
                w = np.logaddexp(w, k) 
            #total = parts[0] + reduce(sum, parts[1:])   
            I[symbol] = w
    return I


def outside_algorithm(forest: CFG, tsort: list, edge_weights: dict, inside: dict, root: Symbol) -> dict:
    """Returns the outside weight of each node"""
    O = dict()
    for symbol in tsort:
        O[symbol] = 0.0
    O[root] = 1.0
    for symbol in reversed(tsort):
        incoming = forest.get(symbol)
        for edge in incoming:
            for u in edge.rhs: # u in tail(e)
                k = edge_weights[edge] * O[symbol]
                for s in edge.rhs:
                    if not u == s:
                        k *= inside[s] # TODO: change to log-sum-exp
                O[u] += k
    return O


def outside_algorithm_log(forest: CFG, tsort: list, edge_weights: dict, inside: dict, root: Symbol) -> dict:
    """Returns the outside weight of each node"""
    O = dict()
    for symbol in tsort:
        # O[symbol] = 0.0
        O[symbol] = -np.inf
    O[root] = 0.0
    for symbol in reversed(tsort):
        incoming = forest.get(symbol)
        for edge in incoming:
            for u in edge.rhs: # u in tail(e)
                k = edge_weights[edge] + O[symbol]
                for s in edge.rhs:
                    if not u == s:
                        k += inside[s] # TODO: change to log-sum-exp
                O[u] = np.logaddexp(O[u], k)
    return O


def viterbi(forest: CFG, tsort: list, edge_weights: dict, inside: dict, root: Symbol) -> dict:
    """Returns the viterbi decoding of hypergraph"""
    Q = deque([root])
    V = list()
    while Q:
        symbol = Q.popleft()
        incoming = forest.get(symbol)
        weights = [0.0]*len(incoming)
        for i, edge in enumerate(incoming):
            weights[i] = edge_weights[edge]
            for u in edge.rhs: # u in tail(e)
                weights[i] *= inside[u] # TODO: change to log-sum-exp
        weight, selected = max(zip(weights, incoming), key=lambda xy: xy[0])
        for sym in selected.rhs:
            if not sym.is_terminal():
                Q.append(sym)
        V.append(selected)    
    return V


def viterbi_log(forest: CFG, tsort: list, edge_weights: dict, inside: dict, root: Symbol) -> dict:
    """Returns the viterbi decoding of hypergraph"""
    Q = deque([root])
    V = list()
    while Q:
        symbol = Q.popleft()
        incoming = forest.get(symbol)
        weights = [1.0]*len(incoming)
        for i, edge in enumerate(incoming):
            weights[i] = np.exp(edge_weights[edge])
            for u in edge.rhs: # u in tail(e)
                weights[i] += inside[u] # TODO: change to log-sum-exp
        weight, selected = max(zip(weights, incoming), key=lambda xy: xy[0])
        for sym in selected.rhs:
            if not sym.is_terminal():
                Q.append(sym)
        V.append(selected)    
    return V


def ancestral_sample(num_samples: int, forest: CFG, tsort: list, edge_weights: dict, inside: dict, root: Symbol) -> dict:
    """Returns the viterbi decoding of hypergraph"""
    samples = list()
    for i in range(num_samples):
        Q = deque([root])
        S = list()
        while Q:
            symbol = Q.popleft()
            incoming = forest.get(symbol)
            weights = [0.0]*len(incoming)
            for i, edge in enumerate(incoming):
                weights[i] = edge_weights[edge]
                for u in edge.rhs: # u in tail(e)
                    weights[i] *= inside[u] # TODO: change to log-sum-exp
            probs = np.array(weights) / sum(weights)
            index = np.argmax(np.random.multinomial(1, probs))
            selected = incoming[index]
            for sym in selected.rhs:
                if not sym.is_terminal():
                    Q.append(sym)
            S.append(selected)    
        samples.append(S)
    # hack since list is unhashable type, so we cannot use Counter (bummer)
    ys = [write_derrivation(d).pop() for d in samples] 
    most_y, counts = Counter(ys).most_common(1)[0]
    dic = {y: d for y, d in zip(ys, samples)}
    most_sampled = dic[most_y]
    return most_sampled, counts


