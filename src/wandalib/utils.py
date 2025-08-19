
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import shutil
from dataclasses import dataclass
import pywanda
import os
import inspect

class AllowedProperties(Enum):
    ROUGHNESS = "Wall roughness"
    FLOW = "Initial delivery rate"

@dataclass
class Scenario:
    scenario_name: str
    parameters: dict
    
# Example of Scenario
#     scenario_name="CLOSURE_VALVE", 
#     parameters={
#     "VALVE V1": {
#         "Action table": assign_closing_time(20)
#     }}
#     ), 
#                  Scenario(
#     scenario_name="CLOSURE_FARM_VALVE", 
#     parameters={
#     "VALVE V2": {
#         "Action table": assign_closing_time(64)
#     }}
#     )
#     ]
# create_scenarios(wanda_file, transient_Scenario, wanda_bin, isUnsteady=True)

def get_all_components_in_model(wanda_model: pywanda.WandaModel) -> list[str]:
    return wanda_model.get_all_components_str()

def get_all_components_dict(wanda_model: pywanda.WandaModel) -> dict[str, list[str]]:
    all_components = get_all_components_in_model(wanda_model)
    components = {}
    for component in all_components:
        component_splited = component.split(" ", 1)
        type_ = component_splited[0]  # Avoid using 'type' as it's a built-in function
        name = component_splited[1]
        # Initialize the list if the key does not exist
        if type_ not in components:
            components[type_] = []
        components[type_].append(name)
    return components

def get_components_from_type(all_components: dict, type:str) -> list[str]:
    return all_components[type]

def show_components_from_type_str(filter: str, all_components: dict) -> str:
    components_list = get_components_from_type(all_components, filter)
    result_str = ""
    for component in components_list:
        result_str += filter + " " + component + "\n"
    return result_str

#################################
    
def create_wanda_model(wanda_file: str, wanda_bin: str) -> tuple[pywanda.WandaModel, str]:
    wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
    wanda_name = os.path.splitext(os.path.basename(wanda_file))[0]
    return wanda_model, wanda_name


def check_if_element_exist(component: str, all_elements: dict[str, list[str]]) -> bool:
    splited_str = component.split()
    component_type = splited_str[0]
    component_name = ' '.join(splited_str[1:])
    if component_type in all_elements and component_name in all_elements[component_type]:
        return True
    else:
        return False
    
def create_scenarios(wanda_file: str, scenarios: list[Scenario], wanda_bin: str, isUnsteady: bool = False)-> None:
    if isUnsteady == True:
        results_dir = "transient_results"
    else:
        results_dir = "steady_results"
    cwd = os.path.dirname(wanda_file)
    try:
        folder_path = os.path.join(cwd, results_dir)
        os.mkdir(folder_path)
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
        print(f"Scenario %s created in path %s from wandabin %s" % (scenario.scenario_name, scenario_path, wanda_bin))
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
            print(new_wanda_model)
            new_wanda_model.run_steady()
            new_wanda_model.run_unsteady()
        else:        
            new_wanda_model.run_steady()
        print("Scenario ran")        
        new_wanda_model.close()
        

def assign_closing_time(closing_time: int, offset_time: int = 10):
    time = [0, offset_time, closing_time + offset_time]
    position = [1, 1, 0]
    return [time, position]
    
def assing_value(wanda_model, component, parameter, value):
    component = wanda_model.get_component(component)
    flow_rate = component.get_property(parameter)
    flow_rate.set_scalar(value/3600)