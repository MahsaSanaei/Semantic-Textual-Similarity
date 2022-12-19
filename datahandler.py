import pickle
import codecs
from pathlib import Path
import pandas as pd
import json

class DataHandler:
    """
        DataHandler class for loading and saving datas and files
    """
    def __init__(self):
        """
            init method
        """

    @staticmethod
    def load_pkl(path: Path):
        """
            loading pickle files
        :param path:
        :return:
        """
        with codecs.open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def load_csv(path: Path, columns: list=None) -> pd:
        """
            loading a csv file
        :param path:
        :return:
        """
        if columns is None:
            data_frame = pd.read_csv(path, sep='\t')
        else:
            data_frame = pd.read_csv(path, names=columns, sep='\t')
        return data_frame

    @staticmethod
    def load_json(path: Path) -> json:
        """
            loading a json file
        :param path:
        :return:
        """
        with codecs.open(path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        return json_data

    
    @staticmethod
    def write_pkl(data, path: Path):
        """
            write to pickle file
        :param data:
        :param path:
        :return:
        """
    
    @staticmethod
    def write_csv(data: pd, path: Path):
        """
            Write CSV file
        :param data: csv data
        :param path: to save pandas file
        :return:
        """
        data_frame = pd.DataFrame(data=data)
        data_frame.to_csv(path, index=False)

    @staticmethod
    def write_json(data:dict, path: Path):
        """
            Write json file
        :param data: json data
        :param path: to to save json file
        :return:
        """
        with codecs.open(path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

