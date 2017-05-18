import lib.formal
import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import parse_forests, make_lexicon, make_lexicon_ALT
from features import featurize_edges, get_full_fset

lexicon = defaultdict(set)
lexicon['le'].update(['the', '-EPS-'])  # we will assume that `le` can be deleted
lexicon['-EPS-'].update(['a', 'the'])  # we will assume that `the` and `a` can be inserted
lexicon['et'].add('and')
lexicon['chien'].add('dog')
lexicon['chat'].add('cat')
lexicon['noir'].update(['black', 'dark'])  
lexicon['blanc'].add('white')
lexicon['rouge'].add('red')
lexicon['petit'].update(['small', 'little'])
lexicon['petite'].update(['small', 'little'])

sents = [('le chien noir', 'the black dog'),
		 ('le petit chien', 'the small dog'),
		 ('le petit chat', 'the small cat'),
		 ('le petit chien noir', 'the little black dog'),
		 ('le chat rouge', 'the red cat'),
		 ('le chien rouge', 'the red dog'),
		 ('le chat', 'the cat'),
		 ('le chien', 'the dog'),
		 ('rouge', 'red')]

# make parses of sents based on lexicon

parses = [parse_forests(fr, en, lexicon) for fr, en in sents]

# make alternative parses

def make_ALT_lexicon(ALT_lexicons):
	ALT_lexicon = defaultdict(set)
	for lexicon in ALT_lexicons:
		for fr, translations in lexicon.items():
			ALT_lexicon[fr].update(translations)
	return ALT_lexicon

ALT_lexicons = [make_lexicon_ALT(fr, en) for fr, en in sents]

ALT_lexicon = make_ALT_lexicon(ALT_lexicons)

ALT_parses = [parse_forests(fr, en, ALT_lexicon) for fr, en in sents]

# make a random translations dictionary

def make_random_transdicts(lexicon):
	fr_en = defaultdict(lambda:defaultdict(float))
	en_fr = defaultdict(lambda:defaultdict(float))
	for fr, translations in lexicon.items():
		for en in translations:
			fr_en[fr][en] = np.random.uniform()
			fr_en[en][fr] = np.random.uniform()
	return fr_en, en_fr

fr_en, en_fr = make_random_transdicts(lexicon)
ALT_fr_en, ALT_en_fr = make_random_transdicts(ALT_lexicon)

# get the full feature set of target_forest and ref_forest together
fset = get_full_fset(parses, fr_en, en_fr)
ALT_fset = get_full_fset(ALT_parses, ALT_fr_en, ALT_en_fr)

print(len(fset))
print(len(ALT_fset))


# w_init = defaultdict(float)
# for feature in fset:
#     w_init[feature] = 0.00001*np.random.uniform()

# w_test, delta_ws = sgd_func_minibatch(5, 0.01, w_init, minibatch=parses, 
#                                       sparse=True, bar=False, log=True, check_convergence=True)


w_init = defaultdict(float)
for feature in ALT_fset:
    w_init[feature] = 1e-5*np.random.uniform()

w_test, delta_ws = sgd_func_minibatch(5, 1e-8, w_init, minibatch=ALT_parses, 
                                      sparse=True, bar=False, log=True, check_convergence=True)




