import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
import dill



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def model_trainer(X_train, y_train, models, params):
    try:
        score_report = {}
        params_report = {}

        for model_name, model_instance in models.items():

            para = params[model_name]
            gs = GridSearchCV(model_instance, param_grid=para, cv=3, scoring='accuracy') 
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            model_score = gs.best_score_

            score_report[model_name] = model_score
            params_report[model_name] = best_params
            
        return score_report, params_report
    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)