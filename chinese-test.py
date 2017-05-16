import libitg as libitg
import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import parse_forests, make_lexicon, make_lexicon_ALT, translations, translations_ALT, read_data
from features import featurize_edges, get_full_fset

def make_total_lexicon(lexicons):
	full_lexicon = defaultdict(set)
	for lexicon in lexicons:
		for src, translations in lexicon.items():
			full_lexicon[src].update(translations)
	return full_lexicon

# get translations
# ch_en, en_ch, _, _ = translations(path='data/lexicon', k=3, null=3, remove_punct=True)
ch_en, en_ch, _, _ = translations_ALT(path='data/lexicon', k=3, null=3, remove_punct=True)

# load corpus of 10 sentences
# corpus = read_data(max_sents=300)
corpus = read_data(max_sents=100) # worked great with init=1e-5 en lr=1e-9
# get only short sentences for ease of training
short_corpus = [(ch, en) for ch, en in corpus if len(en.split()) < 5]
print('\n'.join(map(str,short_corpus)))
print(len(short_corpus))

### PARSES ###

# make lexicons for sentences in short corpus
lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in short_corpus]
# make one big lexicon covering all
lexicon = make_total_lexicon(lexicons)
for k, v in lexicon.items():
	print(k,v)

# make parses based on lexicon
parses = [parse_forests(ch, en, lexicon) for ch, en in short_corpus]


### ALT PARSES ###s

# make lexicons for sentences in short corpus
ALT_lexicons = [make_lexicon_ALT(ch_sent, en_sent) for ch_sent, en_sent in short_corpus]
# make one big lexicon covering all
ALT_lexicon = make_total_lexicon(ALT_lexicons)
# for k, v in ALT_lexicon.items():
# 	print(k, v)

# make parses based on lexicon
ALT_parses = [parse_forests(ch, en, ALT_lexicon) for ch, en in short_corpus]



# get the full feature set of target_forest and ref_forest together
fset = get_full_fset(parses, ch_en, en_ch, sparse=True)
ALT_fset = get_full_fset(ALT_parses, ch_en, en_ch, sparse=True)

print(len(fset))
print(len(ALT_fset))

# Works for corpus 100

w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-5*np.random.uniform()

w_first, delta_ws = sgd_func_minibatch(1, 1e-9, w_init, minibatch=parses, 
                                      sparse=True, bar=False, log=True, check_convergence=True)

w_test, delta_ws = sgd_func_minibatch(5, 1, w_first[-1], minibatch=parses, 
                                      sparse=True, bar=False, log=True, check_convergence=True)

# Works for corpus 300

# w_init = defaultdict(float)
# for feature in fset:
#     w_init[feature] = 1e-9*np.random.uniform()

# w_first, delta_ws = sgd_func_minibatch(1, 1e-9, w_init, minibatch=parses, 
#                                       sparse=True, bar=False, log=True, check_convergence=True)

# w_test, delta_ws = sgd_func_minibatch(5, 1e-2, w_first[-1], minibatch=parses, 
#                                       sparse=True, bar=False, log=True, check_convergence=True)









