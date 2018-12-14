# Image-Caption Retrieval

Junyang Qian & Giacomo Lamberti

(CS229 Machine Learning)

The repository contains:

  - code: directory containing all the modules of the code:
  
          1. main.py       -> call the trainer
          
          2. trainer.py    -> train the models
          
          3. datasource.py -> define dataset as iterator
          
          4. dataset.py    -> load the dataset
          
          5. evaluation.py -> compute the evaluation metrics
          
          
  - results: directory containing text files with a summary of the various results.
          
          1. results_glove_mse_margin_01.txt      -> summary of training of GRU+GloVe model
          
          2. results_glove_LSTM_mse_margin_01.txt -> summary of training of LSTM+GloVe model
          
          3. acc_len.txt                          -> R@10 vs length of the caption for 3 models
  
The dataset (1 GB), including 10-crop VGG19 features, can be downloaded by running:

wget http://www.cs.toronto.edu/~vendrov/order/coco.zip

The code can be run by going to the code directory and typing: python main.py

However, the code won't run without data.
