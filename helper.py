import numpy as np
import pandas as pd

def helper(data):
    print(data.info(), "\n")
    print('missing values')
    print(data.isnull().sum(), "\n")
    print('duplicated rows')
    print(data.duplicated().sum(), "\n")
    data = data.drop_duplicates(keep='first')
    print('data shape')
    print(data.shape, "\n\n")
    print("Number of each label: \n", data['label'].value_counts())
