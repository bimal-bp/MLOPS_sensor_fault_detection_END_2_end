from src.configuration.mongodb_connect import MongoDBClient
from src.utils.exception import customexception 
import os 
import sys 
from src.utils.logger import logging 
from src.pipeline import training_pipeline
from src.pipeline.training_pipeline import TrainPipeline
from src.utils.main_utils import read_yaml_file 
from src.constant.training_pipilien import SAVED_MODEL_DIR
from fastapi import FastAPI
from src.constant.application import APP_HOST,APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import app_run 
from fastapi.responses import Response
from src.ml.model.estimator import ModelResolver,TargetValueMapping
from src.utils.main_utils import load_object
from src.middleware.cors import CORSMiddleware 


env_file_path = os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):
    env_config= read_yaml_file(env_file_path)
    os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']

app= FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credntials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/doc")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Traing pipileine is running already ")
        train_pipeline.run_pipeline()
        return Response("Training succesful !! ")
    except Exception as e:
        return Response(f"Error occured {e}")

@app.get("/predict")
async def predict_route():
    try:
        df = None
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exist():
            return Response("Model Is not Avaliable ")
        best_model_path = model_resolver.get_best_model_path()
        model= load_object(file_path=best_model_path)
        y_pred=model.predict(df)
        df['predicted_column'] =y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
    except Exception as e:
        raise Response(f"Error ouucred {e}")

def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ =="__main__":
    app_run(app,host=APP_HOST,port = APP_PORT)

    