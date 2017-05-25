from util import save_weights, load_weights
from processing import *
from graph import *
from features import weight_function
import progressbar

ch_en, en_ch, _, _ = translations(path='data/lexicon', k=2, null=2, remove_punct=True)

corpus = read_data(max_sents=200)
# corpus = read_data_dev(max_sents=200)
corpus = [(ch, en) for ch, en in corpus if len(en.split()) < 10]


lexicons = [make_lexicon(ch_sent, ch_en) for ch_sent, _ in corpus]
# make one big lexicon covering all words
lexicon = make_total_lexicon(lexicons)


f = open('prediction/ibm1/training/ibm1-prediction.txt', 'w')
for k, (ch, _) in enumerate(corpus):
	ch_sent = ch.split()
	en = []
	for ch_car in ch_sent:
		lex = lexicons[k][ch_car]
		if lex: # crappy shit due to lexicon always containing -EPS-
			en_car = lex.pop()
			if en_car == '-EPS-' and lex:
				en_car = lex.pop()
				if en_car == '-EPS-':
					en_car = ''
			else:
				en_car == ''
		else:
			en_car = ''

		en.append(en_car)
	en_sent = ' '.join(en)
	# print(en_sent)
	f.write(en_sent + '\n')

f.close()
