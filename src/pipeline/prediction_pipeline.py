import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info('Entering prediction')
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,ph: float,Hardness: float,Solids: float,Chloramines: float,Sulfate: float,Conductivity: float,Organic_carbon: float,Trihalomethanes: float,Turbidity: float):
        self.ph = ph
        self.Hardness = Hardness
        self.Solids= Solids
        self.Chloramines= Chloramines
        self.Sulfate= Sulfate
        self.Conductivity= Conductivity
        self.Organic_carbon= Organic_carbon
        self.Trihalomethanes= Trihalomethanes
        self.Turbidity= Turbidity

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "ph": [self.ph], 
                "Hardness": [self.Hardness],
                "Solids": [self.Solids],
                "Chloramines": [self.Chloramines],
                "Sulfate": [self.Sulfate],
                "Conductivity": [self.Conductivity],
                "Organic_carbon": [self.Organic_carbon],
                "Trihalomethanes": [self.Trihalomethanes], 
                "Turbidity": [self.Turbidity]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)