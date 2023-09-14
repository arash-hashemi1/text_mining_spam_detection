import numpy as np
import pandas as pd
import joblib
import re
from data_loader import load_data
from feature_addition import add_feature
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pandas.plotting import table 

## list of models
model_list = ['logistic_regression',
              'svm',
              'multinomial_nb'
]

vect = joblib.load('models/vect.sav') ## vectorizer loaded
model_name = []
model_accuracy = []
model_auc = []
for name in model_list:
    print(f'Loading {name} model...')
    model = joblib.load(f'models/{name}.sav')
    X_train, X_test, y_train, y_test = load_data('data/spam.csv')
    X_test_vectorized = vect.transform(X_test)

    ## creating document length, number of digits, and number of non-word characters for new features
    len_test = X_test.apply(lambda x: len(x))
    num_digit_test = X_test.apply(lambda x: sum(c.isdigit() for c in x))
    num_nonword_test = X_test.apply(lambda x: len(re.findall(r'\W', x)))

    add_ft_test = np.vstack((len_test, num_digit_test, num_nonword_test))

    ## feature addition
    X_test_additional = add_feature(X_test_vectorized, add_ft_test)

    ## model prediction, accuracy, and AUC
    y_pred = model.predict(X_test_additional)
    model_name.append(name)
    model_accuracy.append(model.score(X_test_additional, y_test))
    model_auc.append(roc_auc_score(y_test, model.predict_proba(X_test_additional)[:,1]))

## model comparison
model_comparison = pd.DataFrame({'model': model_name, 'accuracy': model_accuracy, 'AUC': model_auc})
model_comparison.set_index('model', inplace=True)
print(model_comparison)
## save model comparison
model_comparison.to_csv(index=True, path_or_buf='model_comparison.csv')


