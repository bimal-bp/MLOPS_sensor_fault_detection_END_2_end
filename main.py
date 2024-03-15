import os 
import sys 
from src.utils.exception import customexception
from src.utils.logger import logging
from src.pipeline.training_pipeline import TrainPipeline

def start_training():
    try:
        train_pipeline=TrainPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        raise customexception(e,sys)
if __name__ == "__main__":
    start_training()