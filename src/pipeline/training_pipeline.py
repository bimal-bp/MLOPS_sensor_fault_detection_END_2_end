from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
import os ,sys
from src.utils.logger import logging
from src.utils.exception import customexception 
from src.components.data_ingestion import DataIngestion 


class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info('Start data ingestion')
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"data ingestion complet :{data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise customexception(e,sys)


    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running=True

            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()

        except Exception as e:
            raise customexception(e,sys)