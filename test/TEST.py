
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import shutil
from dataclasses import dataclass
import pywanda
import os
import inspect
import wandalib

cwd =  r'C:\Users\juan.guerrero\Juan\dev\wandalib\wanda_model_template_example\wanda_file'
wanda_bin = r'C:\Program Files (x86)\Deltares\Wanda 4.7\Bin\\'
wanda_file = os.path.join(cwd, "Example_Model.wdi")
wanda_basemodel = pywanda.WandaModel(wanda_file, wanda_bin)
# Si se corrió el modelo y se quiere cargar el resultado
wanda_basemodel.reload_output()


all_components = wandalib.get_all_components_dict(wanda_basemodel)

PIPE = ["PIPE LINE START 1", "PIPE LINE START 2", "PIPE LINE START 3", "PIPE LINE DELIVERY1 1", "PIPE LINE DELIVERY1 2"]

checked = wandalib.check_if_element_exist("PIPE LINE START 1", all_components)
print(f"Element exists: {checked}")

from wandalib import Scenario, assign_closing_time

transient_scenarios =[
    Scenario(
        scenario_name="CLOSURE_STANDARD1_VALVE",
        parameters={
            "VALVE STANDARD 1": {
                "Action table": assign_closing_time(20)
                }
            }
        ), 
    # Scenario(
    #     scenario_name="PUMP_TRIP",
    #     parameters={
    #         "VALVE FARM": {
    #             "Action table": assign_closing_time(64)
    #             }
    #         }
    #     )
    ]

# import os
i = False
# Eliminar archivos que comienzan con "NEW_SCENARIO"
if i == True:
    dir = os.path.join(cwd, "transient_results")
    for filename in os.listdir(dir):
        if filename.startswith("NEW_FILE"):
            file_path = os.path.join(dir, filename)
            try:
                os.remove(file_path)
                print(f"Archivo eliminado: {file_path}")
            except Exception as e:
                print(f"No se pudo eliminar {file_path}: {e}")




cwd = os.path.dirname(wanda_file)
print(f"Current working directory: {cwd}")

scenario_name = "NEW_SCENARIO.wdi"
scenario_skeleton = "NEW_SCENARIO.wdx"

scenario_path = os.path.join(cwd, scenario_name)
print(wanda_file)
print(scenario_path)
scenario_skeleton_path = os.path.join(cwd, scenario_skeleton)
# print(mother_case_skeleton)
print(scenario_skeleton_path)
wanda_file_skeleton = os.path.join(cwd, "Example_Model.wdx")
shutil.copy(wanda_file, scenario_path)
shutil.copy(wanda_file_skeleton, scenario_skeleton_path)

wanda_file_copy = pywanda.WandaModel(scenario_path, wanda_bin)
print(wandalib.get_all_components_dict(wanda_file_copy))
component = wanda_file_copy.get_component("VALVE STANDARD 1")
print(dir(component))
print(component.get_name())
print(component.get_name_prefix())
print(component.get_comp_type())
print(component.get_all_properties())
print(component.get_class_name())
# property = component.get_property("Action table")
# print(dir(property))
# table = property.get_table()
# print(dir(table))
# print(table.get_float_data())
# print(assign_closing_time(20))
# table.set_float_data([[0.0, 5.0], [1.0, 1.0]])
property = component.get_property("Inner diameter")
print(dir(property))
print(property.get_scalar_float())
property.set_scalar(0.5)
# print(assign_closing_time(20))
# table.set_float_data([[0.0, 5.0], [1.0, 1.0]])
wanda_file_copy.run_steady()
# wanda_file_copy.run_unsteady()
wanda_file_copy.close()

