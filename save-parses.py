import libitg as libitg
import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset

# ########################################################################
# Saving the parses of a small number given a lexicon, which we also save.
# ########################################################################


#### CORPUS AND TRANSLATIONS ####

# get translations
ch_en, en_ch, _, _ = translations(path='data/lexicon', k=3, null=3, remove_punct=True)
# ch_en, en_ch, _, _ = translations_ALT(path='data/lexicon', k=3, null=3, remove_punct=True)

# load corpus
corpus = read_data(max_sents=300)
# corpus = read_data(max_sents=100) # worked great with init=1e-5 en lr=1e-9

# get only short sentences for ease of training
short_corpus = [(ch, en) for ch, en in corpus if len(en.split()) < 5]
print('\n'.join(map(str,short_corpus)))
print(len(short_corpus))



#### PARSES ####

# make lexicons for sentences in short corpus
lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in short_corpus]
# make one big lexicon covering all words
lexicon = make_total_lexicon(lexicons)
for k, v in lexicon.items():
	print(k,v)

# make parses based on lexicon
parses = [parse_forests(ch, en, lexicon) for ch, en in short_corpus]

# get the full feature set of target_forest and ref_forest together
fset = get_full_fset(parses, ch_en, en_ch, sparse=True)
print(len(fset))
print('\n'.join(sorted(list(fset))))


#### SAVING ####

savepath = '../parses/'

save_parses_separate(short_corpus, lexicon, savepath)

save_lexicon(lexicon, savepath)

save_featureset(fset, savepath)

