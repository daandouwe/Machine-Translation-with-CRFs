class Symbol:
    """
    A symbol in a grammar. In this class we basically wrap a certain type of object and treat it as a symbol.
    """
    
    def __init__(self):
        pass
    
    def is_terminal(self) -> bool:
        """Whether or not this is a terminal symbol"""
        pass

    def root(self) -> Symbol:
        """Some symbols are represented as a hierarchy of symbols, this method returns the root of that hierarchy."""
        pass    
    
    def obj(self) -> object:
        """Returns the underlying python object."""
        pass
    
    def translate(self, target) -> Symbol:
        """Translate the underlying python object of the root symbol and return a new Symbol"""
        pass
    
class Terminal(Symbol):
    """
    Terminal symbols are words in a vocabulary.
    """
    
    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Terminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol
        
    def is_terminal(self):
        return True
        
    def root(self) -> Terminal:
        # Terminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target) -> Terminal:
        return Terminal(target)
        
    def __str__(self):
        return "'%s'" % self._symbol
    
    def __repr__(self):
        return 'Terminal(%r)' % self._symbol
    
    def __hash__(self):
        return hash(self._symbol)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol
    
    def __ne__(self, other):
        return not (self == other)
    
class Nonterminal(Symbol):
    """
    Nonterminal symbols are variables in a grammar.
    """
    
    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Nonterminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol
        
    def is_terminal(self):
        return False
        
    def root(self) -> Nonterminal:
        # Nonterminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target) -> Nonterminal:
        return Nonterminal(target)
    
    def __str__(self):
        return "[%s]" % self._symbol
    
    def __repr__(self):
        return 'Nonterminal(%r)' % self._symbol
    
    def __hash__(self):
        return hash(self._symbol)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol
    
    def __ne__(self, other):
        return not (self == other)
    