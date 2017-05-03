from symbol import Symbol
from span import Span
from rule import Rule
from cfg import CFG
from itg import read_lexicon, make_source_side_itg
from fsa import FSA
from item import Item
from collections import defaultdict

"""
An agenda of active/passive items in CKY/Ealery program.
"""

from collections import defaultdict, deque

class Agenda:

    def __init__(self):
        # we are organising active items in a stack (last in first out)
        self._active = deque([])
        # an item should never queue twice, thus we will manage a set of items which we have already seen
        self._seen = set()
        # we organise incomplete items by the symbols they wait for at a certain position
        # that is, if the key is a pair (Y, i)
        # the value is a set of items of the form
        # [X -> alpha * Y beta, [...i]]
        self._incomplete = defaultdict(set)
        # we organise complete items by their LHS symbol spanning from a certain position
        # if the key is a pair (X, i)
        # then the value is a set of items of the form
        # [X -> gamma *, [i ... j]]
        self._complete = defaultdict(set)
        # here we store the destinations already discovered
        self._destinations = defaultdict(set)

    def __len__(self):
        """return the number of active items"""
        return len(self._active)

    def push(self, item: Item):
        """push an item into the queue of active items"""
        if item not in self._seen:  # if an item has been seen before, we simply ignore it
            self._active.append(item)
            self._seen.add(item)
            return True
        return False

    def pop(self) -> Item:
        """pop an active item"""
        assert len(self._active) > 0, 'I have no items left.'
        return self._active.pop()

    def make_passive(self, item: Item):
        """Store an item as passive: complete items are part of the chart, incomplete items are waiting for completion."""
        if item.is_complete():  # complete items offer a way to rewrite a certain LHS from a certain position
            self._complete[(item.lhs, item.start)].add(item)
            self._destinations[(item.lhs, item.start)].add(item.dot)
        else:  # incomplete items are waiting for the completion of the symbol to the right of the dot
            self._incomplete[(item.next, item.dot)].add(item)

    def waiting(self, symbol: Symbol, dot: int) -> set:
        """return items waiting for `symbol` spanning from `dot`"""
        return self._incomplete.get((symbol, dot), set())

    def complete(self, lhs: Symbol, start: int) -> set:
        """return complete items whose LHS symbol is `lhs` spanning from `start`"""
        return self._complete.get((lhs, start), set())
    
    def destinations(self, lhs: Symbol, start: int) -> set:
        """return destinations (in the FSA) for `lhs` spanning from `start`"""
        return self._destinations.get((lhs, start), set())

    def itercomplete(self):
        """an iterator over complete items in arbitrary order"""
        for items in self._complete.itervalues():
            for item in items:
                yield item
