# second_ex_deep_learning
Window-based Tagging

Peleg shefi and Daniel bazar
316523638 314708181

running part 1
--------------
run tagger1.py

running part 2
--------------
run top_k.py

for requirments, see notes

running part 3
--------------
run tagger1.py -p

for requirments, see notes

running part 4
--------------
run tagger1.py -s (optional also with -p)

for requirments, see notes

running part 5
--------------
run tagger1.py -c (optional also with -p)

for requirments, see notes

CLI usage
---------
 ```console
python .\tagger1.py --help


usage: tagger1.py [-h] [-p] [-s | -c]

A window-based tagger

optional arguments:
  -h, --help        show this help message and exit
  -p, --pretrained  run in pretrained mode
  -s, --subword     run in subword mode
  -c, --cnn         run in cnn mode

if no arguments pass: run in default mode. no special embedding method, just A simple        
window-based tagger
 ```
 
 tuning parameters
 -----------------
 to fit the model to the data, with different parameters, call the function 'parameters_search'. the function take the following arguments:
* params_dict - Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
* n_eopchs - number of epoch to run per configuration
* train_dataset
* dev_dataset
* vocab
* return_best_epoch - [True | False] weather to return in addition the best model at the best epoch
* optimize - Return the best configuration based on the specified [accuracy | loss]
* mission - ['NER' | 'POS']
* pre_trained_emb - pre-trained embedding matrix (work only in pretrained mode)
* pre_vocab - prefix sub-word vocabulary (work only in subword mode)
* suf_vocab - suffix sub-word vocabulary (work only in subword mode)
* cnn_vocab - Convolution-based sub-word vocabulary (work only in cnn mode)
* n_filters - number of filters in cnn (work only in cnn mode)

Notes:
------
for all parts, The working directory should contain:
1) The POS data directory with train, dev and test files.
2) The NER data directory with train, dev and test files.
3) utility.py

In addition, for part 2,3,4,5 that using pre-trained embedding, working directory should contain:
vocab.txt
wordVectors.txt

output graphs are in the "graphs" directory
