# Machine-Translation-with-CRFs
Project 2 of [NLP2](https://uva-slpl.github.io/nlp2/). Read the [project description](readings/project2.pdf) or the [paper](readings/Blunsom08.pdf) that partly inspired it.

## How-to

* Use `save-parses.py` to save the parse-forest of a number of sentence pairs of a corpus. In `translations` you can set `k` and `null` to control how many translations (`k`) and insertions (`null`) to make. Set the size of the corpus in `read_data` and the maximal sentence length just below.

* Use `train.py` to load these parses and train on them. Specify how many sentences you load in the list comprehension in line 11. You can pre-train a `w` and save this. Then reload this one each time you train (redo this whenever you reload new parses!). Use sgd_minibatches when training on a large corpus. 

* Use `predict.py` to load in a trained weights vector `w` and some `parses` in the right format, and predict the best translations (viterbi and sampled). Write these to a prediction .txt file in the folder `predict`. These can be used to compute BLEU scores.

* Note that we're using a hack to prevent huge values in the weight vector: in SGD the optional parameter scale_weight can be set to any integer `k`. Then all values in the weight vector are scaled so that none exceed `10**k`.

* SGD has been updated so that we scale the learning rate each time we make a weight-vector update (i.e. each minibatch). See section 5.2 of [this paper](readings/bottou-sgd-tricks-2012.pdf) on SGD-tricks. This introduces a new hyperparameter `lmbda` which controls the rate of scaling. We now start with a high learning rate of around 1 to 10, and let the formula scale this down.

## Some notes on training

* Use a large batch size. Probably in the range `30-100`. This gives stability to the updates of `w`, since most of the features don't 'fire' for one training example.

* Using shuffle=True we reshuffle the parses and partition these into new minibatches at each iteration. This drastically improves ?movement?. Compare the sentences in [shuffle](prediction/2k/shuffle) to those in [no-shuffle](prediction/2k/no-shuffle) and see the difference: the the `no-shuffle` sentences are almost stationary after the first iteration except for some insertions of ?the?; the `shuffle` sentences continue to change drastically each iteration. I think our best shot is with ?shuffle? for this reason: we just need to take this ?movement? behaviour into account (see note below).

* When we shuffle, we should let the learning rate decay more rapidly. For example `delta_0=10` and `lmbda=50`. Then we start large, but decay rapidly.

## TODO

* Mess around training on different sizes of corpus, with different mini-batch sizes, learning-rates, scale_weights, and regularizers.

* Think about more features. Check `simple_features` for what we already have. Perhaps we need more span features. I added skip-bigram features recently: `le * noir` for the word `chien` in `le chien noir.`, and `chien * -END-` for `.`. 

* Use the script - which we will be provided with soon - to compute the BLEU score of our predicted translations produced in `predict.py` compared with gold-translations provided in the folder [data](data/dev1.zh-en). (Note: we get a number of gold-standard translations - apparently BLEU then works better). We need to write some small script to do this.

## Some issues/questions

* The problem with derivations for which the `p(y,d|x) = nan` is this: the weights vector `w`. This *still* occurs, even with the above described hack. It *only* occurs with long sentences though. I think because for a long sentence, the derivation has many edges. And then `sum([estimated_weights[edge] for edge in derrivation])` gets upset, which we use in `join_prob` to compute the  `p(y,d|x)`. NOTE: This is not *really* an isse: we still get Viterbi estimates! We just cannot compute the correct probability.

* Not sure about the regularizer. Should we use: $\mathbb{E}[D(x,y)] - \mathbb{E}[D_n(x)] + \lambda ||w|| $ or  $\mathbb{E}[D(x,y)] - \mathbb{E}[D_n(x)] - lambda ||w||$?
