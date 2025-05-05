import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wandalib.utils import AllowedProperties


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