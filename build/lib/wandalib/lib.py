import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import shutil
from dataclasses import dataclass
import pywanda
import os
# from dotenv import load_dotenv

# load_dotenv()
# ! DONT TOUCH
# cwd =  r'C:\Users\juan.guerrero\Downloads\wanda\Current Model\MODEL 22_02_2024\Transient\wandalib'
# wanda_bin = os.getenv('WANDA_BIN') 
# print(wanda_bin)
# wanda_name = r'WANDA_MODEL.wdi'
# wanda_file = os.path.join(cwd, wanda_name)

class AllowedProperties(Enum):
    ROUGHNESS = "Wall roughness"
    FLOW = "Initial delivery rate"

@dataclass
class Scenario:
    scenario_name: str
    parameters: dict

# SCENARIO_LIST = [Scenario(
#     scenario_name="CLOSURE_MOMRAH_VALVE", 
#     parameters={
#     "VALVE MOMRAH": {
#         "Action table": assign_closing_time(20)
#     }}
#     ), 
#                  Scenario(
#     scenario_name="CLOSURE_FARM_VALVE", 
#     parameters={
#     "VALVE FARM": {
#         "Action table": assign_closing_time(64)
#     }}
#     )
#                  ]

# create_scenarios(wanda_file, transient_Scenario, wanda_bin, isUnsteady=True)
    
def create_dict_from_list(items):
    result_dict = {}
    for item in items:
        key, value = item.split(' ', 1)
        if key in result_dict:
            result_dict[key].append(value)
        else:
            result_dict[key] = [value]
    return result_dict

    
def create_wanda_model(wanda_file: str, wanda_bin: str):
    wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
    wanda_name = os.path.splitext(os.path.basename(wanda_file))[0]
    return wanda_model, wanda_name
    

def get_all_elements(wanda_model: pywanda.WandaModel):
    element_list = wanda_model.get_all_components_str()
    element_dict = create_dict_from_list(element_list)
    return element_dict


def check_if_element_exist(component: str, all_elements: list):
    splited_str = component.split()
    component_type = splited_str[0]
    component_name = ' '.join(splited_str[1:])
    print(component_type)
    print(component_name)
    if component_type in all_elements and component_name in all_elements[component_type]:
        return True
    else:
        return False
    
def create_scenarios(wanda_file: str, scenarios: list[Scenario], wanda_bin: str, isUnsteady: bool = False):
    results_dir = "transient_results"
    # Create the directory
    cwd = os.path.dirname(wanda_file)
    try:
        os.mkdir(cwd + results_dir)
        print(f"Directory '{results_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{results_dir}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{results_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    mother_case_skeleton = os.path.join(cwd, os.path.splitext(os.path.basename(wanda_file))[0] + ".wdx")
    for scenario in scenarios:
        scenario_wdi = scenario.scenario_name + ".wdi"
        scenario_wdx = scenario.scenario_name + ".wdx"
        scenario_path = os.path.join(cwd, results_dir, scenario_wdi)
        print(wanda_file)
        print(scenario_path)
        scenario_skeleton_path = os.path.join(cwd, results_dir, scenario_wdx)
        print(mother_case_skeleton)
        print(scenario_skeleton_path)
        shutil.copy(wanda_file, scenario_path)
        shutil.copy(mother_case_skeleton, scenario_skeleton_path)
        new_wanda_model = pywanda.WandaModel(scenario_path, wanda_bin)
        for parameter in scenario.parameters:
            # TODO parameterType = parameter.split()[0]
            # if parameterType == "Signal":
            #     signal = new_wanda_model.get_signal_line(parameter)
            if parameter == "GLOBAL PROPERTY":
                for key, value in scenario.parameters[parameter].items():
                    property = new_wanda_model.get_property(key)
                    property.set_scalar(value)
            elif parameter == "SIGNAL DISUSE":
                for signal in scenario.parameters[parameter]:
                    signal_node = new_wanda_model.get_signal_line(f"Signal {signal}")
                    signal_node.set_disused(True)
            else:
                for key, value in scenario.parameters[parameter].items():
                    # print(get_all_elements(new_wanda_model))
                    component = new_wanda_model.get_component(parameter)
                    property = component.get_property(key)
                    if key == "Action table":
                        table = property.get_table()
                        table.set_float_data(value)
                        continue
                    property.set_scalar(value)
        print(f"Scenario %s created in path %s" % (scenario.scenario_name, cwd))        
        print("Running scenario...")
        if isUnsteady == True:
            new_wanda_model.run_unsteady()
        else:        
            new_wanda_model.run_steady()
        print("Scenario ran")        
        new_wanda_model.close()
        
    
# HACK .get_all_properties() for a component and .get_all_components() for a WandaModel
# print(get_all_elements(wanda_model))
# print(wanda_name)
# print(wanda_model)
# check_if_element_exist("VALVE BA 3", get_all_elements(wanda_model))
# check_if_element_exist("VALVE BA 87", get_all_elements(wanda_model))

def assign_closing_time(closing_time: int, offset_time: int = 10):
    time = [0, offset_time, closing_time + offset_time]
    position = [1, 1, 0]
    return [time, position]


    
def assing_value(wanda_model, component, parameter, value):
    component = wanda_model.get_component(component)
    flow_rate = component.get_property(parameter)
    flow_rate.set_scalar(value/3600)
        
def get_node_pressure_steady(wanda_model, node):   
    node = wanda_model.get_node(node)
    wanda_model.read_node_output(node)
    node_pressure = node.get_property("Pressure").get_scalar_float() / 100000
    return node_pressure

# UPDATED
    
def get_transient_pressure_df(wanda_model, pipes, downsampling_factor=1, print_messages = True, is_returning_series = False):
    """
    Genera series de presión para tuberías en un modelo Wanda.

    Args:
        wanda_model: Modelo Wanda que contiene los componentes de las tuberías.
        pipes (list): Lista de nombres de tuberías a procesar.
        downsampling_factor (int): Factor de muestreo para reducir el tamaño de las series. 
                                   El valor por defecto es 1 (sin reducción).
        print_messages (Bool): Printea los valores máximos y mínimos de presión en la tubería en el código
        is_returning_serie (Bool): Devuelve los valores de presion tres series de pandas en lugar de dataframe

    Returns:
        tuple: Series de presión estacionaria, mínima y máxima, como pandas.Series.
    """
    # time_steps = wanda_model.get_time_steps()
    length_steps = []
    steady_pressures = []
    max_pressures = []
    min_pressures = []
    for pipe in pipes:
        
        component = wanda_model.get_component(pipe)
        pressure_data = component.get_property("Pressure")
        profile_data = component.get_property("Profile").get_table().get_float_column("X-distance")

        pressure_series = np.array(pressure_data.get_series_pipe()) /100000
        steady_pressures.append(pressure_series[:, 0])
        
        max_pressures.append(np.array(pressure_data.get_extr_max_pipe()) / 100000)
        min_pressures.append(np.array(pressure_data.get_extr_min_pipe()) / 100000)
        
        length_steps.append(np.linspace(profile_data[0], profile_data[-1], len(pressure_series)))
        
        min_pressure = min(pressure_data.get_extr_min_pipe()) / 100000
        max_pressure = max(pressure_data.get_extr_max_pipe()) / 100000
        if print_messages:
            print(f"For pipeline '{component.get_name()}', min pressure: {min_pressure} bar, max pressure: {max_pressure} bar")

    # Convert lists to numpy arrays
    steady_pressures = np.concatenate(steady_pressures)
    max_pressures = np.concatenate(max_pressures)
    min_pressures = np.concatenate(min_pressures)
    length_steps = np.concatenate(length_steps)
    
    if downsampling_factor > 1:
        steady_pressures = steady_pressures[::downsampling_factor]
        max_pressures = max_pressures[::downsampling_factor]
        min_pressures = min_pressures[::downsampling_factor]
        length_steps = length_steps[::downsampling_factor]
        
    if is_returning_series:
        steady_series = pd.Series(steady_pressures, index=length_steps)
        max_series = pd.Series(max_pressures, index=length_steps)
        min_series = pd.Series(min_pressures, index=length_steps)
        
        return steady_series, min_series, max_series
    
    results_dic = {
        'Steady Pressure': steady_pressures,
        'Maximum Pressure': max_pressures,
        'Minimum Pressure': min_pressures,
    }
    
    results_data = pd.DataFrame(results_dic, index=length_steps)
    results_data.index.name = 'Distance (m)'
    
    return results_data

# TODO PENDIENTE

def get_steady_pipe_results(wanda_model, pipes, downsampling_factor=1, print_messages = True, is_returning_series = False):
    """
    Genera series de presión para tuberías en un modelo Wanda.

    Args:
        wanda_model: Modelo Wanda que contiene los componentes de las tuberías.
        pipes (list): Lista de nombres de tuberías a procesar.
        downsampling_factor (int): Factor de muestreo para reducir el tamaño de las series. 
                                   El valor por defecto es 1 (sin reducción).
        print_messages (Bool): Printea los valores máximos y mínimos de presión en la tubería en el código
        is_returning_serie (Bool): Devuelve los valores de presion tres series de pandas en lugar de dataframe

    Returns:
        tuple: Series de presión estacionaria, mínima y máxima, como pandas.Series.
    """
    # time_steps = wanda_model.get_time_steps()
    length_steps = []
    steady_pressures = []
    max_pressures = []
    min_pressures = []
    for pipe in pipes:
        
        component = wanda_model.get_component(pipe)
        pressure_data = component.get_property("Pressure")
        profile_data = component.get_property("Profile").get_table().get_float_column("X-distance")

        pressure_series = np.array(pressure_data.get_series_pipe()) /100000
        steady_pressures.append(pressure_series[:, 0])
        
        max_pressures.append(np.array(pressure_data.get_extr_max_pipe()) / 100000)
        min_pressures.append(np.array(pressure_data.get_extr_min_pipe()) / 100000)
        
        length_steps.append(np.linspace(profile_data[0], profile_data[-1], len(pressure_series)))
        
        min_pressure = min(pressure_data.get_extr_min_pipe()) / 100000
        max_pressure = max(pressure_data.get_extr_max_pipe()) / 100000
        if print_messages:
            print(f"For pipeline '{component.get_name()}', min pressure: {min_pressure} bar, max pressure: {max_pressure} bar")

    # Convert lists to numpy arrays
    steady_pressures = np.concatenate(steady_pressures)
    max_pressures = np.concatenate(max_pressures)
    min_pressures = np.concatenate(min_pressures)
    length_steps = np.concatenate(length_steps)
    
    if downsampling_factor > 1:
        steady_pressures = steady_pressures[::downsampling_factor]
        max_pressures = max_pressures[::downsampling_factor]
        min_pressures = min_pressures[::downsampling_factor]
        length_steps = length_steps[::downsampling_factor]
        
    if is_returning_series:
        steady_series = pd.Series(steady_pressures, index=length_steps)
        max_series = pd.Series(max_pressures, index=length_steps)
        min_series = pd.Series(min_pressures, index=length_steps)
        
        return steady_series, min_series, max_series
    
    results_dic = {
        'Steady Pressure': steady_pressures,
        'Maximum Pressure': max_pressures,
        'Minimum Pressure': min_pressures,
    }
    
    results_data = pd.DataFrame(results_dic, index=length_steps)
    results_data.index.name = 'Distance (m)'
    
    return results_data

def graph_transient_pressures():
    return 0

def get_surge_vessel_serie(wanda_model, sv):
    time_steps = wanda_model.get_time_steps()
    component = wanda_model.get_component(sv)
    liquid_vol = np.array(component.get_property("Liquid volume").get_series())
    liquid_vol_serie = pd.Series(liquid_vol, index=time_steps)
    print("The minimum volume for the surge vessel ", component.get_name(), "is: ", min(liquid_vol_serie))
    print("The maximum volume for the surge vessel ", component.get_name(), "is: ", max(liquid_vol_serie))
    return liquid_vol_serie

    
def get_pipe_pressure_graphs(wanda_model, pipes, downsampling_factor = 1):

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
    plt.title("Pipeline Pressure")
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [barg]")
    plt.grid()

    plt.legend()
    plt.show()
    


def get_pipe_head_steady(wanda_model, pipes, downsampling_factor=1, is_relative=False):
    """
    Calculate and plot the steady-state head profile for a pipeline system.

    Args:
        wanda_model: The WANDA model object.
        pipes (list): List of pipe names to analyze.
        downsampling_factor (int, optional): Factor to downsample the data. Defaults to 1.
        is_relative (bool, optional): If True, treats pipe distances as relative. Defaults to False.

    Returns:
        list: A list containing the matplotlib figure and axis objects for the plot.
    """
    # Initialize lists to store profile and head data
    profile_x_values = []
    profile_y_values = []
    head_steady_values = []
    length_steps = []
    last_x = 0 
    
    
    for pipe in pipes:
        component = wanda_model.get_component(pipe)
        profile_data = component.get_property("Profile").get_table().get_float_data()
        profile_x = np.array(profile_data[0])  # X-distance data
        profile_y = np.array(profile_data[1])  # Height data
        
        # Get steady-state head data
        pressure_pipe = component.get_property("Head")
        series_pipe = np.array(pressure_pipe.get_series_pipe())
        steady = series_pipe[:, 0]
        
        if is_relative:
            # For the first pipe, initialize the arrays
            if pipe == pipes[0]:
                profile_x_values = profile_x
                profile_y_values = profile_y
            else:
                # Extend x and y arrays for subsequent pipes (skip the first point to avoid overlap)
                updated_distance = profile_x[1:] + last_x
                profile_x_values = np.concatenate([profile_x_values, updated_distance])
                profile_y_values = np.concatenate([profile_y_values, profile_y[1:]])
            steps = np.linspace(profile_x[0], profile_x[-1], len(steady)) + last_x
            last_x = profile_x_values[-1]  # Update the last x-distance
        else:
            profile_x_values.extend(profile_x)
            profile_y_values.extend(profile_y)
            steps = np.linspace(profile_x[0], profile_x[-1], len(steady))

        length_steps = np.concatenate([length_steps, steps])
        head_steady_values.append(steady)

    # Convert lists to numpy arrays
    head_steady_values = np.concatenate(head_steady_values)
    profile_x_values = np.array(profile_x_values)

    # Downsample data if required
    if downsampling_factor != 1:
        head_steady_values = head_steady_values[::downsampling_factor]
        profile_x_values = profile_x_values[::downsampling_factor]

    # Create pandas Series for plotting
    steady_curve = pd.Series(head_steady_values, index=length_steps)
    profile = pd.Series(profile_y_values, index=profile_x_values)
    
    # Plot the profile and steady-state head

    return steady_curve, profile

def get_pipe_head_graphs_from_steady(steady_curve, profile):
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="green")
    bx.plot(steady_curve, label="Steady State Head", color="orange")
    bx.set(xlim=(0, profile.index[-1]))

    # Add plot details
    plt.title("Pipeline Head")
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()    
    # plt.show() 
    return [fig, bx]
    
    
def get_pipe_head_graphs(wanda_model, pipes, downsampling_factor = 1):
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
        
        # print("For pipeline ", component.get_name(), "the minimum head is: ", min(pressure_pipe.get_extr_min_pipe()))
        # print("For pipeline ", component.get_name(), "the maximum head is: ", min(pressure_pipe.get_extr_max_pipe()))

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
    # print(df.keys()) 
    # print(df.columns)
    bx.plot(df, label= "Steady State head", color="orange")
    bx.plot(max_head, label="Maximum head", color="red", linestyle="dashdot")
    bx.plot(min_head, label="Minimum head", color="blue", linestyle="dashdot")
    bx.set(xlim=(0, profile_x[-1]))
    # Add vertical lines
    plt.title("Pipeline Head")
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()
    plt.show()

def get_pipe_head_graphs_from_steady(steady_curve, profile):
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="green")
    bx.plot(steady_curve, label="Steady State Head", color="orange")
    bx.set(xlim=(0, profile.index[-1]))

    # Add plot details
    plt.title("Pipeline Head")
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()    
    # plt.show() 
    return [fig, bx]
    
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
    
def get_pipe_pressure_graphs_w_minrough(wanda_model, pipes):

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
    
    # Create pandas Series
    df = pd.Series(pressure_steady_values, index=len_steps)
    min_pressure = pd.Series(pressure_min_values, index=len_steps)
    max_pressure = pd.Series(pressure_max_values, index=len_steps)
    
    """
    To get an excel with all the time and pressures if is needed:
    df = pd.DataFrame(index=len_steps, columns=wanda_model.get_time_steps())
    for i in range(len(series)):
        df.iloc[i] = series[i] / 100000
    """
    fig, bx = plt.subplots()
    # print(df.keys()) 
    # print(df.columns)
    bx.plot(max_pressure, label="Maximum Pressure")
    bx.plot(min_pressure, label="Minimum Pressure")
    bx.plot(df[0], label= "Steady State Pressure")
    bx.set(xlim=(0, profile_x[-1]))
    bx.minorticks_on()
    plt.title("Pipeline Pressure")
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [barg]")
    plt.grid()

    plt.legend()
    plt.show()
    
def get_profile(wanda_model, pipes):
    profile_x_values = []
    profile_y_values = []

    for pipe in pipes:
        component = wanda_model.get_component(pipe)
        profile_x = component.get_property("Profile").get_table().get_float_column("X-distance")
        profile_y = component.get_property("Profile").get_table().get_float_column("Height")
        profile_x_values.extend(profile_x)
        profile_y_values.extend(profile_y)

    profile = pd.Series(profile_y_values, index=profile_x_values)
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="g")
    
    bx.set(xlim=(0, profile_x_values[-1]))

    # Add vertical lines
    vertical_line_positions = [37864]  # Example positions, adjust as needed
    for pos in vertical_line_positions:
        bx.axvline(x=pos, color='grey', linestyle='--')

    # Add mini-titles above the plot
    bx.text((37864 / 2), max(profile_y_values) + 10, "PIPE MIRFA - BAB", horizontalalignment='center')
    bx.text((37864 + 34945/2), max(profile_y_values) + 10, "PIPE BAB - BU HASA", horizontalalignment='center')
    bx.minorticks_on()
    # Adjust the space to make room for the titles
    plt.subplots_adjust(top=0.85)  # Adjust this value as needed

    plt.title("Pipeline Profile", pad=20)  # Adjust pad to add space between the title and the plot

    plt.xlabel("Distance [m]")
    plt.ylabel("Elevation [m]")
    fig.set_figwidth(8)
    plt.grid()
    plt.legend()
    plt.show()

def get_max_min_prv_pipes(wanda_model, elements):
    time_steps = wanda_model.get_time_steps()
    for el in elements:
        fig, ax = plt.subplots()
        fig, ax2 = plt.subplots()
        fig, bx = plt.subplots()
        component = wanda_model.get_component(el)
        pressure1 = np.array(component.get_property("Pressure 1").get_series()) / 100000
        pressure2 = np.array(component.get_property("Pressure 2").get_series()) / 100000
        discharge1 = np.array(component.get_property("Discharge 1").get_series()) * 3600
        P1 = pd.Series(pressure1, index=time_steps)
        P2 = pd.Series(pressure2, index=time_steps)
        D = pd.Series(discharge1, index=time_steps)
        print("For the element: ", component.get_name())
        print("For before the minimum pressure is: ", min(pressure1))
        print("For before the maximum pressure is: ", max(pressure1))
        print("For after the minimum pressure is: ", min(pressure2))
        print("For after the maximum pressure is: ", max(pressure2))
        print("For before the minimum flow is: ", min(discharge1))
        print("For before the maximum flow is: ", max(discharge1))
        ax.plot(P1, label="Pressure Before Pipe")
        ax2.plot(P2, label="Pressure After Pipe")
        ax.set_title(el)
        ax2.set_title(el)
        ax.grid()
        ax2.grid()
        ax.legend()
        ax2.legend()
        ax.set_xlabel("Time [s]")
        ax2.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [barg]")
        ax2.set_ylabel("Pressure [barg]")
        bx.plot(D)
        bx.set_title(el)
        bx.grid()
        bx.legend()
        bx.set_xlabel("Time [s]")
        bx.set_ylabel("Discharge [m3/hr]")
        bx.set(xlim=(0, time_steps[-1]))
        
        
def get_info_nodes(wanda_model, nodes, title=False):
    fig, bx = plt.subplots()
    time_steps = wanda_model.get_time_steps()
    for el in nodes:
        component = wanda_model.get_node(el)
        pressure = np.array(component.get_property("Pressure").get_series()) / 100000
        P1 = pd.Series(pressure, index=time_steps)
        print("For the element: ", component.get_name())
        print("The minimum pressure is: ", min(pressure))
        print("The maximum pressure is: ", max(pressure))
        bx.plot(P1, label="Pressure in Node")
        if title != False and isinstance(title, str) :
            bx.set_title(title)
        else:
            bx.set_title(el)
        bx.grid()
        bx.legend()
        bx.set_xlabel("Time [s]")
        bx.set_ylabel("Pressure [barg]")
        bx.set_xlim(left=0)
        
def get_node_pressure_series(wanda_model, node):    
    time_steps = wanda_model.get_time_steps()
    component = wanda_model.get_node(node)
    pressure = np.array(component.get_property("Pressure").get_series()) / 100000
    pressure_serie = pd.Series(pressure, index=time_steps)
    print("The minimum pressure for node ", component.get_name(), "is: ", min(pressure_serie))
    print("The maximum pressure for node ", component.get_name(), "is: ", max(pressure_serie))
    return pressure_serie

def get_node_pressure_transient(wanda_model, node):   
    node = wanda_model.get_node(node)
    wanda_model.read_node_output(node) #is necessary?
    time_steps = wanda_model.get_time_steps()
    node_pressure = np.array(node.get_property("Pressure").get_series()) / 100000
    node_pressure = pd.Series(node_pressure, index=time_steps)
    return node_pressure
        
    

def get_pressure_valves(wanda_model, valves):
    fig_num = 0
    fig, axs = plt.subplots(1, len(valves))
    fig.set_figwidth(10)
    fig.tight_layout()
    time_steps = wanda_model.get_time_steps()
    for valve in valves:
        fig, bx = plt.subplots()
        component = wanda_model.get_component(valve)
        pressure1 = np.array(component.get_property("Pressure 1").get_series()) / 100000
        pressure2 = np.array(component.get_property("Pressure 2").get_series()) / 100000
        discharge1 = np.array(component.get_property("Discharge 1").get_series()) * 3600
        P1 = pd.Series(pressure1, index=time_steps)
        P2 = pd.Series(pressure2, index=time_steps)
        D = pd.Series(discharge1, index=time_steps)
        print("For before the valve ", component.get_name(), "the minimum pressure is: ", min(pressure1))
        print("For before the valve ", component.get_name(), "the maximum pressure is: ", max(pressure1))
        print("For after the valve ", component.get_name(), "the minimum pressure is: ", min(pressure2))
        print("For after the valve ", component.get_name(), "the maximum pressure is: ", max(pressure2))
        print("The minimum flow is: ", min(discharge1))
        print("The maximum flow is: ", max(discharge1))
        axs[fig_num].plot(P1, label="Pressure Before Pipe")
        axs[fig_num].plot(P2, label="Pressure After Pipe")
        axs[fig_num].set_title(valve)
        axs[fig_num].grid()
        axs[fig_num].legend()
        axs[fig_num].set_xlabel("Time [s]")
        axs[fig_num].set_ylabel("Pressure [barg]")
        fig_num += 1
        bx.plot(D)
        bx.set_title(valve)
        bx.grid()
        bx.legend()
        bx.set_xlabel("Time [s]")
        bx.set_ylabel("Discharge [m3/hr]")
        bx.set(xlim=(0, time_steps[-1]))
def get_minimum_head(wanda_model, head_node, h1, h2, control_node, minimum_pressure):
    
    maxiter = h1  # Initial guess for head (maximum)
    miniter = h2 # Initial guess for head (minimum)

    cached_results = {}  # Store results of previous function calls

    def get_pressure_from_head(n):
        # Check if the result for n is already cached
        if n in cached_results:
            return cached_results[n], n

        component = wanda_model.get_component(f"BOUNDH {head_node}")
        head =  component.get_property("Head at t = 0 [s]")
        head.set_scalar(n)
        wanda_model.run_steady()
        node = wanda_model.get_node(f"H-node {control_node}")
        NODE_PRESSURE = node.get_property("Pressure").get_scalar_float() / 100000

        # Cache the result for future use
        cached_results[n] = NODE_PRESSURE
        print("The return is: ", NODE_PRESSURE, n)
        # print("With head:", head.get_scalar_float(), "the pressure is:", NODE_PRESSURE, "and the difference for min pressure:", NODE_PRESSURE - minimum_pressure)
        return NODE_PRESSURE, n

    tolerance = 0.001  # Tolerance for the root

    while True:
        press_headmax, headmax = get_pressure_from_head(maxiter)
        press_headmin, headmin = get_pressure_from_head(miniter)

        # Check if the function value at the maximum is close to the target
        if press_headmax > 0 and abs(press_headmax - minimum_pressure) < tolerance:
            node_pressure = press_headmax
            head_result = headmax
            break

        # Check if the function value at the minimum is close to the target
        if press_headmin > 0 and abs(press_headmin - minimum_pressure) < tolerance:
            node_pressure = press_headmin
            head_result = headmin
            break

        # Calculate the midpoint and its corresponding function value
        mean = (maxiter + miniter) / 2
        press_mean, headmean = get_pressure_from_head(mean)

        # Adjust the bounds based on the function value at the midpoint
        if press_mean < minimum_pressure:
            miniter = mean
        else:
            maxiter = mean
        print("maxiter", maxiter, "miniter", miniter)
        # Check if the difference between maxiter and miniter is within tolerance
        if maxiter - miniter < tolerance:
            node_pressure = press_mean
            head_result = headmean
            break
        
    print("Result for head = ", head_result, "Pressure in Bu Hasa: ", node_pressure)

# Ejemplo de uso
# get_minimum_head(wanda_model, "B3", 500, 50, "C", 20.1)

    
def change_parameter(wanda_model, elements, parameter: AllowedProperties, value, is_unsteady = False):
    """Docstring

    Currently only working for roughness

    Returns:
    int:Returning value

   """
   
    if parameter == AllowedProperties.ROUGHNESS:
        coef = 1 / 1000
        
    for element in elements:
        component = wanda_model.get_component(element)
        element_parameter = component.get_property(parameter.value)
        element_parameter.set_scalar(value * coef)
        print("Now the component ", component.get_name(), "has a ", parameter.value, " of ", round(element_parameter.get_scalar_float() * (1/coef), 3))
        
    if is_unsteady == True:
        print("Is running transient...")
        wanda_model.run_unsteady()
        print("Done")
    else:
        print("Is running steady...")
        wanda_model.run_steady()
        print("Done")
    print("Closing simulation")
    wanda_model.close()
        
        
        