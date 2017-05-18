# Machine-Translation-with-CRFs
Project 2 of NLP2

## How-to

* Use `save-parses.py` to save the parse-forest of a number of sentence pairs of a corpus. In `translations` you can set `k` and `null` to control how many translations (`k`) and insertions (`null`) to make. Set the size of the corpus in `read_data` and the maximal sentence length just below.

* Use `run-chinese-load.py` to load these parses and train on them. Specify how many sentences you load in the list comprehension in line 11. You can pre-train a `w` and save this. Then reload this one each time you train (redo this whenever you reload new parses!). Use sgd_minibatches when training on a large corpus. 

* Note that we're using a hack to prevent hughe values in the weight vector: in sgd the optional parameter scale_weight can be set to any integer `k`. Then all values in the weight vector are scaled so that none exceed `10**k`.

## Do

* Mess around training on different sizes of corpus, with different mini-batch sizes, learning-rates, scale_weights, and regularizers.

## Some issues/questions

* Not sure about the regularizer. Should we use: \\\mathbb{E}[D(x,y)] - \mathbb{E}[D_n(x)] + \lambda ||w|| \\ or  $\mathbb{E}[D(x,y)] - \mathbb{E}[D_n(x)] - lambda ||w||$?
