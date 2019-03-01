#! загружаємо список серій для тренування
# по 1000 послідовних значень за певний проміжок часу
# і зберігаємо їх
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA
import datetime
import coin_desk_client

#видає серію з 2000 елементів
def get_data():
    series = Series(
        coin_desk_client.get_historical_price('BTC'))
    return series
#print(get_data())
