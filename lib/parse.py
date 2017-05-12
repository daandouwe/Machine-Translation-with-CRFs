from symbol import Symbol
from span import Span
from rule import Rule
from cfg import CFG
from itg import read_lexicon, make_source_side_itg
from fsa import FSA
from item import Item
from agenda import Agenda
from collections import defaultdict

def axioms(cfg: CFG, fsa: FSA, s: Symbol) -> list:
    """
    Axioms for Earley.

    Inference rule:
        -------------------- (S -> alpha) \in R and q0 \in I
        [S -> * alpha, [q0]] 
        
    R is the rule set of the grammar.
    I is the set of initial states of the automaton.

    :param cfg: a CFG
    :param fsa: an FSA
    :param s: the CFG's start symbol (S)
    :returns: a list of items that are Earley axioms  
    """
    items = []
    for q0 in fsa.iterinitial():
        for rule in cfg.get(s):
            items.append(Item(rule, [q0]))
    return items

def predict(cfg: CFG, item: Item) -> list:
    """
    Prediction for Earley.

    Inference rule:
        [X -> alpha * Y beta, [r, ..., s]]
        --------------------   (Y -> gamma) \in R
        [Y -> * gamma, [s]] 
        
    R is the ruleset of the grammar.

    :param item: an active Item
    :returns: a list of predicted Items or None  
    """
    items = []
    for rule in cfg.get(item.next):
        items.append(Item(rule, [item.dot]))
    return items

def scan(fsa: FSA, item: Item, eps_symbol: Terminal=Terminal('-EPS-')) -> list:
    """
    Scan a terminal (compatible with CKY and Earley).

    Inference rule:

        [X -> alpha * x beta, [q, ..., r]]
        ------------------------------------    where (r, x, s) \in FSA and x != \epsilon
        [X -> alpha x * beta, [q, ..., r, s]]
        
        
    If x == \epsilon, we have a different rule
    
        [X -> alpha * \epsilon beta, [q, ..., r]]
        ---------------------------------------------   
        [X -> alpha \epsilon * beta, [q, ..., r, r]]
    
    that is, the dot moves over the empty string and we loop into the same FSA state (r)

    :param item: an active Item
    :param eps_symbol: a list/tuple of terminals (set to None to disable epsilon rules)
    :returns: scanned items
    """
    assert item.next.is_terminal(), 'Only terminal symbols can be scanned, got %s' % item.next
    if eps_symbol and item.next.root() == eps_symbol:
        return [item.advance(item.dot)]
    else:
        destination = fsa.destination(origin=item.dot, label=item.next.root().obj())  # we call .obj() because labels are strings, not Terminals
        if destination < 0:  # cannot scan the symbol from this state
            return []
        return [item.advance(destination)]
        
def complete(agenda: Agenda, item: Item):
    """
    Move dot over nonterminals (compatible with CKY and Earley).

    Inference rule:

        [X -> alpha * Y beta, [i ... k]] [Y -> gamma *, [k ... j]]
        ----------------------------------------------------------
                 [X -> alpha Y * beta, [i ... j]]

    :param item: an active Item.
        if `item` is complete, we advance the dot of incomplete passive items to `item.dot`
        otherwise, we check whether we know a set of positions J = {j1, j2, ..., jN} such that we can
        advance this item's dot to.
    :param agenda: an instance of Agenda
    :returns: a list of items
    """
    items = []
    if item.is_complete():
        # advance the dot for incomplete items waiting for item.lhs spanning from item.start
        for incomplete in agenda.waiting(item.lhs, item.start):
            items.append(incomplete.advance(item.dot))
    else:
        # look for completions of item.next spanning from item.dot
        for destination in agenda.destinations(item.next, item.dot):                
            items.append(item.advance(destination))
    return items
    
def earley(cfg: CFG, fsa: FSA, start_symbol: Symbol, sprime_symbol=None, eps_symbol=Terminal('-EPS-')):
    """
    Earley intersection between a CFG and an FSA.
    
    :param cfg: a grammar or forest
    :param fsa: an acyclic FSA
    :param start_symbol: the grammar/forest start symbol
    :param sprime_symbol: if specified, the resulting forest will have sprime_symbol as its starting symbol
    :param eps_symbol: if not None, the parser will support epsilon rules
    :returns: a CFG object representing the intersection between the cfg and the fsa 
    """
    
    # start an agenda of items
    A = Agenda()
    
    # this is used to avoid a bit of spurious computation
    have_predicted = set()

    # populate the agenda with axioms
    for item in axioms(cfg, fsa, start_symbol):
        A.push(item)
        
    # call inference rules for as long as we have active items in the agenda
    while len(A) > 0:  
        antecedent = A.pop()
        consequents = []
        if antecedent.is_complete():  # dot at the end of rule                    
            # try to complete other items            
            consequents = complete(A, antecedent)
        else:
            if antecedent.next.is_terminal():  # dot before a terminal 
                consequents = scan(fsa, antecedent, eps_symbol=eps_symbol)
            else:  # dot before a nonterminal
                if (antecedent.next, antecedent.dot) not in have_predicted:  # test for spurious computation
                    consequents = predict(cfg, antecedent)  # attempt prediction
                    have_predicted.add((antecedent.next, antecedent.dot))
                else:  # we have already predicted in this context, let's attempt completion
                    consequents = complete(A, antecedent)
        for item in consequents:            
            A.push(item)
        # mark this antecedent as processed
        A.make_passive(antecedent)

    def iter_intersected_rules():
        """
        Here we convert complete items into CFG rules.
        This is a top-down process where we visit complete items at most once.
        """
        
        # in the agenda, items are organised by "context" where a context is a tuple (LHS, start state)
        to_do = deque()  # contexts to be processed
        discovered_set = set()  # contexts discovered
        top_symbols = []  # here we store tuples of the kind (start_symbol, initial state, final state)
        
        # we start with items that rewrite the start_symbol from an initial FSA state
        for q0 in fsa.iterinitial():
            to_do.append((start_symbol, q0))  # let's mark these as discovered
            discovered_set.add((start_symbol, q0))
                        
        # for as long as there are rules to be discovered
        while to_do:
            nonterminal, start = to_do.popleft()                             
            # give every complete item matching the context above a chance to yield a rule
            for item in A.complete(nonterminal, start):
                # create a new LHS symbol based on intersected states
                lhs = Span(item.lhs, item.start, item.dot)
                # if LHS is the start_symbol, then we must respect FSA initial/final states
                # also, we must remember to add a goal rule for this
                if item.lhs == start_symbol:
                    if not (fsa.is_initial(start) and fsa.is_final(item.dot)):
                        continue  # we discard this item because S can only span from initial to final in FSA                        
                    else:
                        top_symbols.append(lhs)
                # create new RHS symbols based on intersected states
                #  and update discovered set
                rhs = []
                for i, sym in enumerate(item.rule.rhs):
                    context = (sym, item.state(i))
                    if not sym.is_terminal() and context not in discovered_set:
                        to_do.append(context)  # book this nonterminal context
                        discovered_set.add(context)  # mark as discovered
                    # create a new RHS symbol based on intersected states
                    rhs.append(Span(sym, item.state(i), item.state(i + 1)))
                yield Rule(lhs, rhs)
        if sprime_symbol:
            for lhs in top_symbols:
                yield Rule(sprime_symbol, [lhs])
    # return the intersected CFG :)
    return CFG(iter_intersected_rules())
