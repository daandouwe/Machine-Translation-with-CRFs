import numpy as np
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset

# ########################################################################
# Saving the parses of a small number given a lexicon, which we also save.
# ########################################################################


#### CORPUS AND TRANSLATIONS ####

# get translations
ch_en, en_ch, _, _ = translations(path='data/lexicon', k=3, null=3, remove_punct=True)

# load corpus
corpus = read_data(max_sents=200)

# get only short sentences for ease of training
corpus = [(ch, en) for ch, en in corpus if len(en.split()) < 3]
print(len(corpus))
# always save all the english sentences for reference (computing the BLEU)
f = open('../parses/eps-200/reference.txt', 'w')
for ch, en in corpus:
	f.write(en + '\n')
f.close()

#### PARSES ####

# make lexicons for sentences in short corpus
lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in corpus]
# make one big lexicon covering all words
lexicon = make_total_lexicon(lexicons)
for k, v in lexicon.items():
	print(k,v)


#### SAVING ####

savepath = '../parses/eps-200/'

fset = save_parses_separate(corpus, lexicon, savepath, ch_en, en_ch, eps=True, sparse=True)

save_lexicon(lexicon, savepath)

save_featureset(fset, savepath)

print(len(fset))
print('\n'.join(sorted(list(fset))))
