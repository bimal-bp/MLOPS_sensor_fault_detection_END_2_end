import os 


TARGET_COLUMN = "class"
PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "sensor.csv"


# Data ingestion part 
DATA_INGESTION_COLLECTION_NAME: str = "sensor"
DATA_INGESTION_DIR_NAME:str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:float = 0.2

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

# Data validation 

DATA_VALIDATION_DIR_NAME: str = "data_validtion"
DATA_VALIDATION_VALID_DIR: str="validated"
DATA_VALIDATON_INVALID_DIR:str= "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str  = "drift_report"
DATA_VALIDATION_DRIFT_FILE_NAME:str ="report.yaml"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"



