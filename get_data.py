file_path="D:/MLOPS_sensor_fault_detection_END_2_end/aps_failure_training_set1.csv"
from src.data_access.sensor_data import SensorData
from src.constant.training_pipilien import DATA_INGESTION_COLLECTION_NAME
from main import set_env_variable 
import os 

if __name__=="__main__":
    data_file_path="D:/MLOPS_sensor_fault_detection_END_2_end/aps_failure_training_set1.csv"
    env_file_path="D:/MLOPS_sensor_fault_detection_END_2_end/env.yaml"
    set_env_variable(env_file_path)
    print(os.environ['MONGO_DB_URL'])
    sd=SensorData()
    if DATA_INGESTION_COLLECTION_NAME in sd.mongo_client.database.list_collection_names():
        sd.mongo_client.database[DATA_INGESTION_COLLECTION_NAME].drop()
    sd.save_csv_file(file_path=data_file_path,collection_name=DATA_INGESTION_COLLECTION_NAME)
