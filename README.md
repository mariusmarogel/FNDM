# FNDM
This repository represents the implementation of 'Context and Network-Aware Fake News Detection and Mitigation'.

- the SparseShield implementation repository can be found [here](https://github.com/DS4AI-UPB/CONTAIN/tree/main/SparseShield_NIvsHS)
- in this repository, we use a file called 't15_imm_nodes.txt' to access the nodes to be immunized from SparseShield algorithm. Each line contains k nodes to be immunized, with k = 10% of the total number of nodes in the graph.

**In order to obtain the desired results, please read the following details.**
## Notebooks
- Word2VecEmbeddings, BERTandBERTweetEmbeddings, Node2VecEmbeddings, DeepWalkEmbeddings. Running these notebooks creates in order: word embedding matrices, BERT and BERTweet text vectorizations (both will be saved in the current folder), DeepWalk node embeddings which will be saved in folders '32d' and '100d', and Node2Vec node embeddings in folders '32d' and '100d' (follow the notebooks to associate filename->embeddings). 
- Word2VecTrainingAndTesting, BERTTrainingAndTesting, BERTweetTraningAndTesting. Running these notebooks creates a results folder with evaluations on using six different RNN-based architectures, with or without Node2Vec and DeepWalk node embeddings. Each of these results will be saved independently.
- T15Node2VecTesting, T16Node2VecTesting, DeepWalkTesting. Running these notebooks creates results of node2vec embeddings with different (p, q, dimensions) and deepwalk embeddings with 32 dimensions.
## Results
- test.py. This Python script uses numpy and a given filename to create an output file of the form: mean(results) +/- std(results), where the mean and the std are computed with numpy built-in methods.
- test_n2v.py. This script has the same functionality, but runs on a different type of result file and can be used for creating the same results but for the node2vec different embeddings.
## Application
- app.py. This is a Flask implementation of visual examples for this paper by running the application and entering the localhost address in a browser. Once connected, you can enter a number (a dataset index) and press the 'Predict' button to run a saved model (bert_n2v_model.h5) and find on the page the original tweet, the model's prediction and two graphs: the original network of interactions with the source tweet and the same graph with highlighted nodes by SparseShield algorithm. It renders an html page found at 'templates/index.html' and a css styling file at 'static/css/style.css' 



