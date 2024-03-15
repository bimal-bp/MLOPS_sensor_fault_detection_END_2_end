import yaml 
from src.utils.exception import customexception 
from src.utils.logger import logging
import os 
import sys 
from pathlib import Path 
import numpy as np 
import dill 


def write_yaml_file(file_path:str,content:object,replace:bool=False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)

    except Exception as e:
        raise customexception(e,sys)


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise customexception(e, sys) from e

def save_object(file_path:str,obj:object) -> None:
    try:
        logging.info('Entered save object')
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path ,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise customexception(e,sys)

def load_object(file_path:str,) -> object:
    try:
        if not os.apth.exists(file_path):
            raise Exception(f"file path not exist :{file_path}")
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise customexception(e,sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise customexception(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise customexception(e, sys) from e
