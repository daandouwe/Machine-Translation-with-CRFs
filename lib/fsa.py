from symbol import Symbol
from span import Span
from rule import Rule
from cfg import CFG
from itg import read_lexicon, make_source_side_itg
from collections import defaultdict


class FSA:
    """
    A container for arcs. This implements a deterministic unweighted FSA.
    """
    
    def __init__(self):
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        self._states = []
        self._initial = set()
        self._final = set()
        
    def nb_states(self):
        """Number of states"""
        return len(self._states)
    
    def nb_arcs(self):
        """Number of arcs"""
        return sum(len(outgoing) for outgoing in self._states)
    
    def add_state(self, initial=False, final=False) -> int:
        """Add a state marking it as initial and/or final and return its 0-based id"""
        sid = len(self._states)
        self._states.append(defaultdict(int))
        if initial:
            self.make_initial(sid)
        if final:
            self.make_final(sid)
        return sid
    
    def add_arc(self, origin, destination, label: str):
        """Add an arc between `origin` and `destination` with a certain label (states should be added before calling this method)"""
        outgoing = self._states[origin]
        outgoing[label] = destination
    
    def destination(self, origin: int, label: str) -> int:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin >= len(self._states):
            return -1
        outgoing = self._states[origin] 
        if not outgoing:
            return -1
        return outgoing.get(label, -1)
    
    def make_initial(self, state: int):
        """Mark a state as initial"""
        self._initial.add(state)
        
    def is_initial(self, state: int) -> bool:
        """Test whether a state is initial"""
        return state in self._initial
        
    def make_final(self, state: int):
        """Mark a state as final/accepting"""
        self._final.add(state)
        
    def is_final(self, state: int) -> bool:
        """Test whether a state is final/accepting"""
        return state in self._final
        
    def iterinitial(self):
        """Iterates over initial states"""
        return iter(self._initial)
    
    def iterfinal(self):
        """Iterates over final states"""
        return iter(self._final)
    
    def iterarcs(self, origin: int):
        return self._states[origin].items() if origin < len(self._states) else []
    
    def __str__(self):
        lines = ['states=%d' % self.nb_states(), 
                 'initial=%s' % ' '.join(str(s) for s in self._initial),
                 'final=%s' % ' '.join(str(s) for s in self._final),
                 'arcs=%d' % self.nb_arcs()]        
        for origin, arcs in enumerate(self._states):
            for label, destination in sorted(arcs.items(), key=lambda pair: pair[1]):            
                lines.append('origin=%d destination=%d label=%s' % (origin, destination, label))
        return '\n'.join(lines)
        
def make_fsa(string: str) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i, word in enumerate(string.split()):
        fsa.add_state()  # create a destination state 
        fsa.add_arc(i, i + 1, word)  # label the arc with the current word
    fsa.make_final(fsa.nb_states() - 1)
    return fsa