import pywanda
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib import *
cwd =  r'C:\Users\juan.guerrero\Downloads\wanda\WAVE V6\transient_results'
wanda_bin = r'C:\Program Files (x86)\Deltares\Wanda 4.7\Bin\\'

pipes = ["PIPE PBAB", "PIPE PBUHASA"]  
valves = ["VALVE BAB ESD", "VALVE BH ESD"]
SV = ["DAMPER MIRFA", "DAMPER BAB", "DAMPER BUHASA", "DAMPER BAB EXTRA"]
NODES = ["H-node DC.a", "H-node CF.a"]

# max_flow = 0
# max_flow_case = ""
# max_volume = 0
# max_volume_case = ""
# max_flow_2 = 0
# max_flow_case_2 = ""
# max_volume_2 = 0
# max_volume_case_2 = ""

# # for i in range(0, 11):
# #     for j in range(1, 4):
# for i in range(10, 11):
#     for j in range(1, 2):
wanda_name = f"WAVE-MODEL-6-2.wdi"
wanda_file = os.path.join(cwd, wanda_name)
wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
for i in range(1, 5):
    air_valve = f"AIRVvn A{i}"
    component = wanda_model.get_component(air_valve)
    print(component.get_property("Chamber area").set_scalar(0.0001))
wanda_model.run_steady()
# print(component.get_scalar_float())
#         print("For model:", wanda_name)
#         print('Changing the ')
#         steady1, min1, max1 = get_pressure_series(wanda_model, pipes) 
