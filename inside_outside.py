import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from collections import defaultdict
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
                        k *= I[s] # TODO: change to log-sum-exp
                O[u] += k
    return O