import pywanda
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math
import sys
import os

# Agregar la ruta de wanda-library al sys.path
ruta_wanda_library = os.path.abspath(os.path.join("..", "wanda-library"))
sys.path.append(ruta_wanda_library)

# Importar el paquete wandalib
from wandalib import *


cwd =  r'C:\Users\juan.guerrero\Downloads\wanda\SIO'
wanda_bin = r'C:\Program Files (x86)\Deltares\Wanda 4.7\Bin\\'

start_line = ["PIPE LINE START 1", "PIPE LINE START 2", "PIPE LINE START 3", "PIPE LINE START 4", "PIPE LINE START 5", "PIPE LINE START 6"]

farm_line = [*start_line, "PIPE LINE FARM 1", "PIPE LINE FARM 2", "PIPE LINE FARM 3", "PIPE LINE FARM 4", "PIPE LINE FARM 5", "PIPE LINE FARM 6"]

momrah_line = [*start_line, "PIPE LINE MOMRAH 1", "PIPE LINE MOMRAH 2", "PIPE LINE MOMRAH 3", "PIPE LINE MOMRAH 4", "PIPE LINE MOMRAH 5"]


reservoir = ["BOUNDH B1"]
tap = ["TAP FARM"]
wanda_file = os.path.join(cwd, "SIO_Basemodel.wdi")
sio_basemodel = pywanda.WandaModel(wanda_file, wanda_bin)
sio_basemodel.reload_output()

reservoir_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]
tap_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]
pipe_parameters = ["Discharge 1", "Head 1", "Pressure 1", "Volume"]



steady_farm, profile_farm = get_head_steady(sio_basemodel, farm_line)
steady_momrah, profile_momrah = get_head_steady(sio_basemodel, momrah_line)

fig, bx = get_pipe_head_graphs_from_steady(steady_farm, profile_farm, title="Pipeline Head to Farms")
fig, ax = get_pipe_head_graphs_from_steady(steady_momrah, profile_momrah, title="Pipeline Head to MOMRAH")

av_start_line_1v = [(390, 2000), (1379.47, 2006)]
av_farm_line_1v = [*av_start_line_1v, (3752.46, 2020)]
av_momrah_line_1v = [*av_start_line_1v, (3135.11, 2023), (3439.37, 2020)]
# Coordenadas donde se colocarán los triángulos (ejemplo)
av_start_line_2v = [(713.55, 2008.2), (2021.43, 2020.22), (2561.34, 2024.29)]
av_farm_line_2v = [*av_start_line_2v, (3261.46, 2023.28), (3835.22, 2021.48)]
av_momrah_line_2v = [*av_start_line_2v]

# Añadir triángulos al gráfico
# Crear gráficos para farm
add_air_valves(bx, av_farm_line_1v, size=35, label="1 Air Valve")
add_air_valves(bx, av_farm_line_2v, size=35, label="2 Air Valves", color="blue")
bx.legend()

# Crear gráficos para momrah
add_air_valves(ax, av_momrah_line_1v, size=35, label="1 Air Valve")
add_air_valves(ax, av_momrah_line_2v, size=35, label="2 Air Valves", color="blue")
ax.legend()
plt.show()

farm_steady, farm_max_press, farm_min_press = get_transient_pressures(sio_basemodel, farm_line, print_messages=False, is_returning_series=True)

farm_transient_df = {
    "Steady Pressure": farm_steady,
    "Maximum Pressure": farm_max_press,
    "Minimum Pressure": farm_min_press
}

farm_transient_df = pd.DataFrame(farm_transient_df)
print(farm_transient_df)
graph_transient_pressures(farm_transient_df, title="Transient Pressures in Farm")

momrah_transient_df = get_transient_pressures(sio_basemodel, momrah_line, print_messages=False)
print(momrah_transient_df)
graph_transient_pressures(momrah_transient_df, title="Transient Pressures in MOMRAH")

get_pipe_head_graphs(sio_basemodel, farm_line, title="Pipeline Head to Farms")
get_pipe_head_graphs(sio_basemodel, momrah_line, title="Pipeline Head to MOMRAH")


component = sio_basemodel.get_component("PIPE LINE START 1")
pressure_line = component.get_property("Pressure")

print(max(pressure_line.get_extr_max_pipe()))
print(pressure_line.get_extr_max_pipe())
# print(pressure_line.get_series_pipe())
print(min(pressure_line.get_extr_min_pipe()))
print(pressure_line.get_extr_min_pipe())



