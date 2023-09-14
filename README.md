# text_mining_spam_detection

This repository trains machine learning models for spam detection tasks:

- Fit a count vectorizer to the training/data to tokenize the raw data.
- Train three ML models for spam detection: logistic regression, support vector machines, and multinomial naive Bayes.
- Evaluate the models on vectorized test data and compare the accuracy and area under the curve.


---
# How to Run
- Clone the repository and `cd` into it.
- Install the requirements by running `pip install -r requirements.txt`.
- In your terminal, run `export PYTHONPATH=$PYTHONPATH:$(pwd)` to add the current directory to your `PYTHONPATH`.
- Run `python src/train.py 'data/spam.csv' <model_name>` to train the selected model and save both the model and vectorizer as <model_name>_trained and 'vect_trained'

### Note: for <model_name>, you should choose one of the following: 'logistic_regression', 'svm', 'multinomial_nb'
- Run 'python src/model_eval.py' to load the provided save models, and create the accuracy and AUC metrics on test data. 
