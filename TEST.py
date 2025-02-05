import pywanda
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math
from lib import * # type: ignore
cwd =  r'C:\Users\juan.guerrero\Downloads\wanda\SIO'
wanda_bin = r'C:\Program Files (x86)\Deltares\Wanda 4.7\Bin\\'

line_momrah = ["PIPE P1", "PIPE P2"]  
line_farma = ["PIPE P1", "PIPE P3", "PIPE P4"]  
reservoir = ["BOUNDH B1"]
tap = ["TAP FARM"]
wanda_file = os.path.join(cwd, "SIO_Basemodel.wdi")
sio_basemodel = pywanda.WandaModel(wanda_file, wanda_bin)
sio_basemodel.reload_output()

reservoir_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]
tap_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]
pipe_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]

steady, profile = get_pipe_head_steady(sio_basemodel, line_farma, is_relative=True)
fig, bx = get_pipe_head_graphs(steady, profile)
steady, profile = get_pipe_head_steady(sio_basemodel, line_momrah, is_relative=True)
fig, bx = get_pipe_head_graphs(steady, profile)
plt.show()