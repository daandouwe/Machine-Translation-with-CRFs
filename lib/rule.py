from symbol import Symbol
from span import Span
from collections import defaultdict

class Rule(object):
    """
    A rule is a container for a LHS symbol and a sequence of RHS symbols.
    """

    def __init__(self, lhs: Symbol, rhs: list):
        """
        A rule takes a LHS symbol and a list/tuple of RHS symbols
        """
        assert isinstance(lhs, Symbol), 'LHS must be an instance of Symbol'
        assert len(rhs) > 0, 'If you want an empty RHS, use an epsilon Terminal'
        assert all(isinstance(s, Symbol) for s in rhs), 'RHS must be a sequence of Symbol objects'
        self._lhs = lhs
        self._rhs = tuple(rhs)

    def __eq__(self, other):
        return type(self) == type(other) and self._lhs == other._lhs and self._rhs == other._rhs

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self._lhs, self._rhs))

    def __str__(self):
        return '%s ||| %s' % (self._lhs, ' '.join(str(s) for s in self._rhs))
    
    def __repr__(self):
        return 'Rule(%r, %r)' % (self._lhs, self._rhs)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs
    
    @property
    def arity(self):
        return len(self._rhs)
    