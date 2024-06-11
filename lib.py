import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum

class AllowedProperties(Enum):
    ROUGHNESS = "Wall roughness"
    
    
def get_pressure_series(wanda_model, pipes):
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
    steady = pd.Series(pressure_steady_values, index=len_steps)
    min_pressure = pd.Series(pressure_min_values, index=len_steps)
    max_pressure = pd.Series(pressure_max_values, index=len_steps)
    
    return steady, min_pressure, max_pressure

def get_surge_vessel_serie(wanda_model, sv):
    time_steps = wanda_model.get_time_steps()
    component = wanda_model.get_component(sv)
    liquid_vol = np.array(component.get_property("Liquid volume").get_series())
    liquid_vol_serie = pd.Series(liquid_vol, index=time_steps)
    print("The minimum volume for the surge vessel ", component.get_name(), "is: ", min(liquid_vol_serie))
    return liquid_vol_serie

    
def get_pipe_pressure_graphs(wanda_model, pipes):

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
    
    
def get_pipe_head_graphs(wanda_model, pipes):
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
    
def get_surge_vessels_info(wanda_model, surge_vessels):
    fig_num = 0
    fig, axs = plt.subplots(1, len(surge_vessels))
    time_steps = wanda_model.get_time_steps()
    for sv in surge_vessels:
        component = wanda_model.get_component(sv)
        liquid_vol = np.array(component.get_property("Liquid volume").get_series())
        liquid_vol_serie = pd.Series(liquid_vol, index=time_steps)
        print("The minimum volume for the surge vessel ", component.get_name(), "is: ", min(liquid_vol_serie))
        axs[fig_num].plot(liquid_vol_serie, label=f"Liquid Volume of SV")
        axs[fig_num].set_title(sv)
        axs[fig_num].grid()
        axs[fig_num].legend()
        axs[fig_num].set_xlabel("Time [s]")
        axs[fig_num].set_ylabel("Liquid Volume [m3]")
        fig_num += 1
    fig.set_figwidth(15)
    fig.tight_layout()
    
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
        bx.set(xlim=(0, 800))
        
        
def get_info_nodes(wanda_model, nodes):
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
        bx.set_title(el)
        bx.grid()
        bx.legend()
        bx.set_xlabel("Time [s]")
        bx.set_ylabel("Pressure [barg]")


        
    

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
        bx.set(xlim=(0, 800))
    
    
    
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
        
        
        