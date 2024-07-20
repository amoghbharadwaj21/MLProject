import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Entered model training method")
            
            X_train, y_train,X_test, y_test = train_data[:,:-1], train_data[:,-1], test_data[:,:-1], test_data[:,-1]
            
            models = {
                "Linear Regression":LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = [key for key in model_report if model_report[key] == best_model_score][0]
            
            best_model = models[best_model_name]
            
            if(best_model_score<0.6):
                raise CustomException("No best model found")
            
            logging.info("Best model found on both train and test data")
            save_object(
                file_path=self.model_trainer_config.trained_model,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2
            
        except Exception as e:
            raise CustomException(e, sys)