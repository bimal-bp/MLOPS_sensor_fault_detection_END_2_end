from src.configuration.mongodb_connect import MongoDBClient
import os 
from src.utils.main_utils import read_yaml_file

if __name__ == '__main__':
    mongodb_client = MongoDBClient()
    print("collection name:",mongodb_client.database.list_collection_names())

env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']
