## *** -- IMPORT DEPENDENCIES -- *** ##
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

## *** ---------------------------------- *** ##

## *** -- Create The SVC Model -- *** ##

def SVC_model():
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    
    clf = GridSearchCV( SVC(), param_grid = tuned_parameters,scoring = 'recall')
    
    
    return clf