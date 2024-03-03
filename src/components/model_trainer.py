import sys
import os
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, model_trainer

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.mode_trianer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                                )
            
            models ={
                    "BaggingClassifier": BaggingClassifier(), 
                    "GradientBoostingClassifier": GradientBoostingClassifier(), 
                    "DecisionTreeClassifier": DecisionTreeClassifier(),
                    "KNeighborsClassifier": KNeighborsClassifier()
                    }
            params = {
                "BaggingClassifier":{
                                        'n_estimators': [10, 50, 100],
                                        'max_samples': [0.5, 0.7, 1.0]
                                    },
                "GradientBoostingClassifier":{
                                        'learning_rate': [0.01, 0.1],
                                        'n_estimators': [50, 100],
                                        'max_depth': [3, 5]
                                    },
                "DecisionTreeClassifier":{
                                        'max_depth': [None, 5, 10],
                                        'min_samples_split': [2, 5],
                                        'min_samples_leaf': [1, 2]
                                    },
                "KNeighborsClassifier":{
                                        'n_neighbors': [3, 5, 7],
                                        'weights': ['uniform', 'distance'],
                                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                                    }
            }
            score_report, params_report = model_trainer(X_train=X_train, y_train=y_train, models = models, params= params)

            ## To Get the Best model score from dict
            max_score = max(sorted(score_report.values()))

            ## To Get the Best model name from dict
            best_model_name = list(params_report.keys())[list(score_report.values()).index(max_score)]
            best_model_params = params_report[best_model_name]
            if max_score < 0.6:
                logging.info("No Best model Found")
                return 0
            
            logging.info(f"Best model on testing dataset")
            model_name = globals()[best_model_name]

            model = model_name(**best_model_params)
            model.fit(X_train, y_train)
            
            ## saving model in artifacts 
            save_object(file_path=self.mode_trianer_config.trained_model_file_path,
                        obj = model)
            predicted = model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            
            
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)