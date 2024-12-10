import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import copy



def create_24_hour_dict():
    bi_dict = {}

    for i in range(24): # creates 24 intervals, one for every hour
        start_time = pd.to_datetime(f"2019-03-31 {i}:00:00.000").time()
        if i == 23:
            end_time = pd.to_datetime(f"2019-03-31 00:00:00.000").time()
        else:
            end_time = pd.to_datetime(f"2019-03-31 {i+1}:00:00.000").time()
        
        bi_dict[(start_time, end_time)] = 0
    return bi_dict