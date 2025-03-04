import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import shutil
from dataclasses import dataclass
import pywanda
import os
import inspect

        


# UPDATED
    




   






    
def get_pipe_pressure_graphs(wanda_model, pipes, title="Pipeline Pressure", downsampling_factor = 1):

    len_steps = []
    pressure_steady_values = []
    pressure_max_values = []
    pressure_min_values = []
    
    for pipe in pipes:
        component = wanda_model.get_component(pipe)

        pressure_pipe = component.get_property("Pressure")
        series_pipe = np.array(pressure_pipe.get_series_pipe()) /100000
        
        steady = series_pipe[:, 0]
        pressure_steady_values.append(steady)
        pressure_max_values.append(np.array(pressure_pipe.get_extr_max_pipe())/100000)
        pressure_min_values.append(np.array(pressure_pipe.get_extr_min_pipe())/100000)
        profile_x = component.get_property("Profile").get_table().get_float_column("X-distance")
        len_steps.append(np.linspace(profile_x[0], profile_x[-1], len(series_pipe)))
        print("For pipeline ", component.get_name(), "the minimum pressure is: ", min(pressure_pipe.get_extr_min_pipe())/100000)
        print("For pipeline ", component.get_name(), "the maximum pressure is: ", max(pressure_pipe.get_extr_max_pipe())/100000)

    # Convert lists to numpy arrays
    pressure_steady_values = np.concatenate(pressure_steady_values)
    pressure_max_values = np.concatenate(pressure_max_values)
    pressure_min_values = np.concatenate(pressure_min_values)
    len_steps = np.concatenate(len_steps)
    
    if downsampling_factor != 1:
        pressure_steady_values = pressure_steady_values[::downsampling_factor]
        pressure_max_values = pressure_max_values[::downsampling_factor]
        pressure_min_values = pressure_min_values[::downsampling_factor]
        len_steps = len_steps[::downsampling_factor]
    
    # Create pandas Series
    df = pd.Series(pressure_steady_values, index=len_steps)
    min_pressure = pd.Series(pressure_min_values, index=len_steps)
    max_pressure = pd.Series(pressure_max_values, index=len_steps)
    

    fig, bx = plt.subplots()
    bx.plot(df, label= "Steady State Pressure", color="orange")
    bx.plot(max_pressure, label="Maximum Pressure", color="red", linestyle="dashdot")
    bx.plot(min_pressure, label="Minimum Pressure", color="blue", linestyle="dashdot")
    bx.set(xlim=(0, profile_x[-1]))
    bx.minorticks_on()
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [barg]")
    plt.grid()

    plt.legend()
    plt.show()
   


    
    
def get_pipe_head_graphs_from_transient(wanda_model, pipes, title="Pipeline Head", downsampling_factor = 1):
    profile_x_values = []
    profile_y_values = []
    len_steps = []
    head_steady_values = []
    head_max_values = []
    head_min_values = []

    for pipe in pipes:
        component = wanda_model.get_component(pipe)
        profile_x = component.get_property("Profile").get_table().get_float_column("X-distance")
        profile_y = component.get_property("Profile").get_table().get_float_column("Height")
        
        profile_x_values.extend(profile_x)
        profile_y_values.extend(profile_y)
        
        pressure_pipe = component.get_property("Head")
        series_pipe = np.array(pressure_pipe.get_series_pipe())
        
        steady = series_pipe[:, 0]
        head_steady_values.append(steady)
        head_max_values.append(np.array(pressure_pipe.get_extr_max_pipe()))
        head_min_values.append(np.array(pressure_pipe.get_extr_min_pipe()))
        
        len_steps.append(np.linspace(profile_x[0], profile_x[-1], len(series_pipe)))
        

    # Convert lists to numpy arrays
    head_steady_values = np.concatenate(head_steady_values)
    head_max_values = np.concatenate(head_max_values)
    head_min_values = np.concatenate(head_min_values)
    len_steps = np.concatenate(len_steps)
    
    if downsampling_factor != 1:
        head_steady_values = head_steady_values[::downsampling_factor]
        head_max_values = head_max_values[::downsampling_factor]
        head_min_values = head_min_values[::downsampling_factor]
        len_steps = len_steps[::downsampling_factor]
    
    # Create pandas Series
    df = pd.Series(head_steady_values, index=len_steps)
    profile = pd.Series(profile_y_values, index=profile_x_values)
    min_head = pd.Series(head_min_values, index=len_steps)
    max_head = pd.Series(head_max_values, index=len_steps)
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="green")
    bx.plot(df, label= "Steady State head", color="orange")
    bx.plot(max_head, label="Maximum head", color="red", linestyle="dashdot")
    bx.plot(min_head, label="Minimum head", color="blue", linestyle="dashdot")
    bx.set(xlim=(0, profile_x[-1]))
    # Add vertical lines
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()
    plt.show()

    
 
def get_surge_vessels_info(wanda_model, surge_vessels):
    fig_num = 0
    num_plots = len(surge_vessels)
    # Asegurar que tengamos una cuadrícula de al menos 2x2
    num_rows = 2
    num_cols = 2 if num_plots > 1 else 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    time_steps = wanda_model.get_time_steps()
    
    # Aplanar el arreglo de ejes si es necesario
    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for sv in surge_vessels:
        component = wanda_model.get_component(sv)
        liquid_vol = np.array(component.get_property("Liquid volume").get_series())
        liquid_vol_serie = pd.Series(liquid_vol, index=time_steps)
        print("The minimum volume for the surge vessel ", component.get_name(), "is: ", min(liquid_vol_serie))
        print("The minimum volume for the surge vessel ", component.get_name(), "is: ", max(liquid_vol_serie))
        axs[fig_num].plot(liquid_vol_serie, label=f"Liquid Volume of SV")
        axs[fig_num].set_title(sv)
        axs[fig_num].grid()
        axs[fig_num].legend()
        axs[fig_num].set_xlabel("Time [s]")
        axs[fig_num].set_ylabel("Liquid Volume [m3]")
        axs[fig_num].set(xlim=(0, time_steps[-1]))
        fig_num += 1

    # Si hay menos subplots que ejes disponibles, eliminar los vacíos
    if fig_num < len(axs):
        for ax in axs[fig_num:]:
            fig.delaxes(ax)

    fig.tight_layout()
    plt.show()

    

    



        
        
        