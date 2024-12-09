import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, x_train, y_train, x_test, y_test, models):
        """
        Evaluate multiple regression models and return a report.
        """
        try:
            report = {}
            for name, model in models.items():
                logging.info(f"Training model: {name}")
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                r2 = r2_score(y_test, y_pred)
                report[name] = r2
            return report
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the train and test arrays")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate models
            logging.info("Evaluating models")
            model_report = self.evaluate_model(x_train, y_train, x_test, y_test, models)

            # Select the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Check if the best model score is acceptable
            if best_model_score < 0.6:
                raise CustomException(
                    f"No suitable model found. Best model {best_model_name} has an R2 score of {best_model_score}"
                )

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Best model saved successfully")
            y_pred=best_model.predict(x_test)
            r2_square=r2_score(y_test,y_pred)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)
