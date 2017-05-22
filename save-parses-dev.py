import numpy as np
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset

# ######################################################################## #
# Saving the parses of some sentences given a lexicon, which we also save. #
# ######################################################################## #


#### corpus an translations ####

# get translations
ch_en, en_ch, _, _ = translations(path='data/lexicon', k=3, null=3, remove_punct=True)

# load corpus
corpus = read_data_dev(max_sents=500)
start = 0
# get only short sentences for ease of training
corpus = [(ch, en) for ch, en in corpus if len(ch.split()) < 50][start:]

print(len(corpus))
# always save all the english sentences for reference (computing the BLEU)
f = open('../parses/dev/reference.txt', 'w')
for ch, en in corpus:
	f.write(en + '\n')
f.close()

#### parses ####

# make lexicons for sentences in short corpus
lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in corpus]
# make one big lexicon covering all words
lexicon = make_total_lexicon(lexicons)
for k, v in lexicon.items():
	print(k,v)


#### saving ####

savepath = '../parses/dev/ml10-3trans/'

fset = save_parses_separate(corpus, lexicon, savepath, ch_en, en_ch, eps=True, sparse=True, start=start)

save_lexicon(lexicon, savepath)

save_featureset(fset, savepath)

print(len(fset))
print('\n'.join(sorted(list(fset))))
