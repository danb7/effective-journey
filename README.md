# second_ex_deep_learning
Window-based Tagging

running part 1
--------------
run tagger1.py.
to fit the model to the data, call the function 'parameters_search'. the function take the following arguments:
* params_dict - Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
* n_eopchs - number of epoch to run per configuration
* train_dataset
* dev_dataset
* optimize - Return the best configuration based on the specified [accuracy | loss]
* mission - ['NER' | 'POS']

The working directory should contain:
1) The POS data directory with train, dev and test files.
2) The NER data directory with train, dev and test files.
3) utility.py

output graphs are in the "graphs" directory
