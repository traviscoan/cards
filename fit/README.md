# Fitting the RoBERTa-Logistic ensemble classifer

## RoBERTa

We used the fantastic `simpletransformers` library to train, test, and perform inference for the RoBERTa side of our model. For more on how to install the `simpletransformers` library, please see:

https://simpletransformers.ai/docs/installation/

The `roberta/` subdirectory contains the code to fit and evaluate the model. Specifically, directory includes the following 3 Jupyter notebooks which walk you through the process:

* `cards_training.ipynb`: Includes the code (and, importantly, the hyperparameters) used to fit the CARDS model.
* `cards_evaluation.ipynb`: Provides the code to evaluate the model performance on held-out data.
* `cards_inference.ipynb`: Provides code to infer classes in unseen data.

## Logistic

