# Machine-Translation-with-CRFs
Project 2 of NLP2

## Some issues/questions
* On training time, should we use `make_lexicon`, or `make_lexicon_ALT`? 

* The parse forests for Chinese-English sentence pair of length 8 is really huge! Pickling this with `save_parses_separate` takes around 22Mb per pair (and we have +- 40.000 pairs..). This is 800Gb! This will not work right?

* Parsing a sentence (especially generating target_forest) takes long. Around 4 minutes... So 4*40.000 = too long?

* Should we use logs somewhere in the `inside` and `outside` algorithms?
