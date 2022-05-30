1 The task1 checkpoint file implements various NLP state of the art models like BERT, Bertweet, RoBERTa with multiple configurations.
The code in the file is self explanatory and comments were added at relevant cells.
General structure of code is: data reading -> loading -> data pre-processing (depends on us) -> 
NLP model transformer -> data encoding -> creating data loader -> NLP model class -> pre trained model import -> freezing the layer -> model training -> evaluation.

2 For Task 2 Upload the data and run the model


3 For task 3 Upload the data and run each of the given files

4. For Task 4 Folder , Follow the Instructions Below
a. for running the BERT-base-uncased Model uncomment the lines 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
transformer = AutoModel.from_pretrained("bert-base-uncased")
b. for running the BERT-large-uncased
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased",do_lower_case=True)
transformer = AutoModel.from_pretrained("bert-large-uncased")
Chnge the shape value to 1024
c. for running the covid-twitter-bert
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert",do_lower_case=True)
transformer = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
change the shape value to 1024
d. for running the scibert
tokenizer = AutoTokenizer.from_pretrained("lordtt13/COVID-SciBERT")
transformer = AutoModel.from_pretrained("lordtt13/COVID-SciBERT")
