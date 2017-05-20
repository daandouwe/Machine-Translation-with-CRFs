import numpy as np
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset




#### CORPUS AND TRANSLATIONS ####

# get translations
ch_en, en_ch, _, _ = translations(path='data/lexicon', k=3, null=3, remove_punct=True)

# load corpus
corpus = read_data(max_sents=5)

#### PARSES ####

lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in corpus]
# make one big lexicon covering all words
lexicon = make_total_lexicon(lexicons)
for k, v in lexicon.items():
	print(k,v)


src_sent = corpus[0][0]
print(src_sent)
tgt_sent = corpus[0][1]
print(tgt_sent)

tgt_forest, ref_forest, src_fsa, tgt_sent = parse_forests_finite(src_sent, tgt_sent, lexicon)

print(tgt_forest)

print(ref_forest)
