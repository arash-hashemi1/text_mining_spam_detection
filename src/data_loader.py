import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """loads the data from path name and creates train and test sets

    Args:
        path: data path to csv file

    Returns:
        X_train, X_test, y_train, y_test: pandas dataframe
    """
    spam_data = pd.read_csv(path)
    spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
    X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)

    return X_train, X_test, y_train, y_test
