from symbol import Symbol
from span import Span
from rule import Rule
from cfg import CFG
from itg import read_lexicon, make_source_side_itg
from fsa import FSA
from collections import defaultdict
"""
An item in a CKY/Earley program.
"""

class Item:
    """A dotted rule used in CKY/Earley where dots store the intersected FSA states."""

    def __init__(self, rule: Rule, dots: list):
        assert len(dots) > 0, 'I do not accept an empty list of dots'
        self._rule = rule
        self._dots = tuple(dots)

    def __eq__(self, other):
        return type(self) == type(other) and self._rule == other._rule and self._dots == other._dots

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return hash((self._rule, self._dots))

    def __repr__(self):
        return '{0} ||| {1}'.format(self._rule, self._dots)

    def __str__(self):
        return '{0} ||| {1}'.format(self._rule, self._dots)

    @property
    def lhs(self) -> Symbol:
        return self._rule.lhs

    @property
    def rule(self) -> Rule:
        return self._rule

    @property
    def dot(self) -> int:
        return self._dots[-1]

    @property
    def start(self) -> int:
        return self._dots[0]

    @property
    def next(self) -> Symbol:
        """return the symbol to the right of the dot (or None, if the item is complete)"""
        if self.is_complete():
            return None
        return self._rule.rhs[len(self._dots) - 1]

    def state(self, i) -> int:
        """The state associated with the ith dot"""
        return self._dots[i]

    def advance(self, dot) -> Item:
        """return a new item with an extended sequence of dots"""
        return Item(self._rule, self._dots + (dot,))

    def is_complete(self) -> bool:
        """complete items are those whose dot reached the end of the RHS sequence"""
        return len(self._rule.rhs) + 1 == len(self._dots)