import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import os
@dataclass

class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','proprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            nums=['writing score','reading score']

            cats=[
                "gender","race/ethnicity","parental level of education","lunch","test preparation course",]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())])
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                    ("scaler",StandardScaler())

                ]
            )
            logging.info("Numerical cols standard scaled")
            logging.info("Cat cols stnadard scaled")
            preprocessor=ColumnTransformer(
                [
                    ("num_piepline",num_pipeline,nums),
                    ("cat_pipeline",cat_pipeline,cats)


                ]
            )
            return preprocessor

        except Exception as e:
                raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_obj()
            target_col="math score"
            nums=["writing score","reading score"]
            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                 input_feature_train_arr,np.array(target_feature_train_df)

            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("saved preprocessing object")
            save_object(
                 file_path=self.data_transformation_config.preprocessor_obj_file,
                 obj=preprocessing_obj
            )
            return (
                 train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file,
                 


            )
        except Exception as e:
            raise CustomException(e,sys)
             
             




                 
                




                
                