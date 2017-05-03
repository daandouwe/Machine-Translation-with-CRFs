from symbol import Symbol

class Span(Symbol):
    """
    A span can be a terminal, a nonterminal, or a span wrapped around two integers.
    Internally, we represent spans with tuples of the kind (symbol, start, end).
    
    Example:
        Span(Terminal('the'), 0, 1)
        Span(Nonterminal('[X]'), 0, 1)
        Span(Span(Terminal('the'), 0, 1), 1, 2)
        Span(Span(Nonterminal('[X]'), 0, 1), 1, 2)
    """
    
    def __init__(self, symbol: Symbol, start: int, end: int):
        assert isinstance(symbol, Symbol), 'A span takes an instance of Symbol, got %s' % type(symbol)
        self._symbol = symbol
        self._start = start
        self._end = end
        
    def is_terminal(self):
        # a span delegates this to an underlying symbol
        return self._symbol.is_terminal()
        
    def root(self) -> Symbol:
        # Spans are hierarchical symbols, thus we delegate 
        return self._symbol.root()
    
    def obj(self) -> (Symbol, int, int):
        """The underlying python tuple (Symbol, start, end)"""
        return (self._symbol, self._start, self._end)
    
    def translate(self, target) -> Span:
        return Span(self._symbol.translate(target), self._start, self._end)
    
    def __str__(self):
        return "%s:%s-%s" % (self._symbol, self._start, self._end)
    
    def __repr__(self):
        return 'Span(%r, %r, %r)' % (self._symbol, self._start, self._end)
    
    def __hash__(self):
        return hash((self._symbol, self._start, self._end))
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol and self._start == other._start and self._end == other._end
    
    def __ne__(self, other):
        return not (self == other)