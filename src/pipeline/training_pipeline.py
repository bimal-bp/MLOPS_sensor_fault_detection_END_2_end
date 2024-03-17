from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
import os ,sys 
from src.utils.logger import logging
from src.utils.exception import customexception 
from src.components.data_ingestion import DataIngestion 
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


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

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise customexception(e, sys)
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation= DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
            )
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise customexception(e,sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running=True
            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
        except Exception as e :
            raise customexception(e,sys)