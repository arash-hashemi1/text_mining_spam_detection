import re
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib
from data_loader import load_data
from feature_addition import add_feature


def model_train(path, model_name):
    """train the model and save the model to models folder

    Args:
        path: data path to csv file
        model_name: name of the model
    """
    X_train, X_test, y_train, y_test = load_data('data/spam.csv')
    X_train_new = X_train[:2000]
    y_train_new = y_train[:2000]
    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train_new)
    X_train_vectorized = vect.transform(X_train_new)

    ## creating document length, number of digits, and number of non-word characters for new features
    len_train = X_train_new.apply(lambda x: len(x))
    num_digit_train = X_train_new.apply(lambda x: sum(c.isdigit() for c in x))
    num_nonword_train = X_train_new.apply(lambda x: len(re.findall(r'\W', x)))

    ## feature addition
    add_ft_train = np.vstack((len_train, num_digit_train, num_nonword_train))
    X_train_additional = add_feature(X_train_vectorized, add_ft_train)

    ## training the appropriate model based on user model choice
    if model_name == 'svm':
        model = SVC(C=10000, probability=True)
    elif model_name == 'multinomial_nb':
        model = MultinomialNB(alpha=0.1)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(C=100, max_iter=1000)
    else:
        print('Please enter a valid model name')

    model.fit(X_train_additional, y_train_new)

    ## saving the model
    file_name = "".join([model_name, '_trained'])
    vect_name = "".join('vect_trained')
    joblib.dump(model, f'models/{file_name}.sav')
    joblib.dump(vect, f'models/{vect_name}.sav')



if __name__ == '__main__':
    model_train(*sys.argv[1:])