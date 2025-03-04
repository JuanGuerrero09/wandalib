import pywanda
import numpy as np
import pandas as pd

# Transient
## Node
def get_node_pressure_steady(wanda_model:pywanda.WandaModel, node: str) -> list[float]:   
    node = wanda_model.get_node(node)
    wanda_model.read_node_output(node)
    node_pressure = node.get_property("Pressure").get_scalar_float() / 100000
    return node_pressure

## Pipeline

### Head
def get_transient_heads(wanda_model: pywanda.WandaModel, pipes: list[str], downsampling_factor: int =1, is_relative: bool =False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calculate and plot the steady-state head profile for a pipeline system.

    Args:
        wanda_model: The WANDA model object.
        pipes (list): List of pipe names to analyze.
        downsampling_factor (int, optional): Factor to downsample the data. Defaults to 1.
        is_returning_series (bool, optional): If True, returns data as Pandas Series. Defaults to False.
        is_relative (bool, optional): If True, treats pipe distances as relative. Defaults to False.

    Returns:
        tuple: A tuple containing the results DataFrame and profile Series, or individual Series if is_returning_series is True.
    """
    # Initialize lists to store profile and head data
    profile_x_values = []
    profile_y_values = []
    head_steady_values = []
    head_max_values = []
    head_min_values = []
    length_steps = []
    last_x = 0 

    for pipe in pipes:
        component = wanda_model.get_component(pipe)
        profile_data = component.get_property("Profile").get_table().get_float_data()
        profile_x = np.array(profile_data[0])  # X-distance data
        profile_y = np.array(profile_data[1])  # Height data

        # Get steady-state head data
        head_pipe = component.get_property("Head")
        steady = np.array(head_pipe.get_series_pipe())


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
        
        head_steady_values.append(steady[:, 0])
        head_max_values.append(np.array(head_pipe.get_extr_max_pipe()))
        head_min_values.append(np.array(head_pipe.get_extr_min_pipe()))

    # Convert lists to numpy arrays
    head_steady_values = np.concatenate(head_steady_values)
    head_max_values = np.concatenate(head_max_values)
    head_min_values = np.concatenate(head_min_values)
    profile_x_values = np.array(profile_x_values)

    # Downsample data if required
    if downsampling_factor != 1:
        head_steady_values = head_steady_values[::downsampling_factor]
        head_max_values = head_max_values[::downsampling_factor]
        head_min_values = head_min_values[::downsampling_factor]
        profile_x_values = profile_x_values[::downsampling_factor]
        length_steps = length_steps[::downsampling_factor]  # Downsample length_steps as well

    # Create profile Series
    profile = pd.Series(profile_y_values, index=profile_x_values)

    # Create results DataFrame
    results_dic = {
        'Steady Head': head_steady_values,
        'Maximum Head': head_max_values,
        'Minimum Head': head_min_values,
    }
    results_data = pd.DataFrame(results_dic, index=length_steps)
    results_data.index.name = 'Distance (m)'

    return results_data, profile

### Pressure

def get_transient_pressures(wanda_model: pywanda.WandaModel, pipes: list[str], downsampling_factor:int =1, print_messages:bool = True) -> pd.DataFrame:
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
    
    results_dic = {
        'Steady Pressure': steady_pressures,
        'Maximum Pressure': max_pressures,
        'Minimum Pressure': min_pressures,
    }
    
    results_data = pd.DataFrame(results_dic, index=length_steps)
    results_data.index.name = 'Distance (m)'
    
    return results_data

## Surge Vessel

def get_surge_vessel_serie(wanda_model, sv):
    time_steps = wanda_model.get_time_steps()
    component = wanda_model.get_component(sv)
    liquid_vol = np.array(component.get_property("Liquid volume").get_series())
    liquid_vol_serie = pd.Series(liquid_vol, index=time_steps)
    print("The minimum volume for the surge vessel ", component.get_name(), "is: ", min(liquid_vol_serie))
    print("The maximum volume for the surge vessel ", component.get_name(), "is: ", max(liquid_vol_serie))
    return liquid_vol_serie

############################################

# Steady

## Pipeline

### Head

def get_head_steady(wanda_model: pywanda.WandaModel, pipes:list[str], downsampling_factor:int =1, is_relative:bool =False):
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

### Pressure
 
def get_pressure_steady(wanda_model: pywanda.WandaModel, pipes: list[str], downsampling_factor:int =1, is_relative:bool =False, show_messages: bool = False) -> pd.Series:
    """
    Calculate and plot the steady-state pressure profile for a pipeline system.

    Args:
        wanda_model: The WANDA model object.
        pipes (list): List of pipe names to analyze.
        downsampling_factor (int, optional): Factor to downsample the data. Defaults to 1.
        is_relative (bool, optional): If True, treats pipe distances as relative. Defaults to False.

    Returns:
        list: A list containing the matplotlib figure and axis objects for the plot.
    """
    # Initialize lists to store profile and pressure data
    pressure_steady_values = []
    len_steps = []
    last_x = 0 
    
    
    for pipe in pipes:
        component = wanda_model.get_component(pipe)
        
        # Get steady-state head data
  
        pressure_pipe = component.get_property("Pressure")
        series_pipe = np.array(pressure_pipe.get_series_pipe())/100000
        steady = series_pipe[:, 0]
        profile_x = component.get_property("Profile").get_table().get_float_column("X-distance")
        profile_x_values = []
        
        if is_relative:
            # For the first pipe, initialize the arrays
            len_steps.append(np.linspace(profile_x[0], profile_x[-1], len(series_pipe)))
            if pipe == pipes[0]:
                profile_x_values = profile_x
            else:
                # Extend x and y arrays for subsequent pipes (skip the first point to avoid overlap)
                updated_distance = profile_x[1:] + last_x
                profile_x_values = np.concatenate([profile_x_values, updated_distance])
            steps = np.linspace(profile_x[0], profile_x[-1], len(steady)) + last_x
            last_x = profile_x_values[-1]  # Update the last x-distance
        else:
            profile_x_values.extend(profile_x)
            steps = np.linspace(profile_x[0], profile_x[-1], len(steady))

        len_steps = np.concatenate([len_steps, steps])
        pressure_steady_values.append(steady)

    # Convert lists to numpy arrays
    pressure_steady_values = np.concatenate(pressure_steady_values)
    profile_x_values = np.array(profile_x_values)

    # Downsample data if required
    if downsampling_factor != 1:
        pressure_steady_values = pressure_steady_values[::downsampling_factor]
        profile_x_values = profile_x_values[::downsampling_factor]

    # Create pandas Series for plotting
    steady_curve = pd.Series(pressure_steady_values, index=len_steps)
    
    if show_messages == True:
        print("For pipeline ", component.get_name(), "the minimum pressure is: ", min(steady_curve))
        print("For pipeline ", component.get_name(), "the maximum pressure is: ", max(steady_curve))
    # Plot the profile and steady-state pressure

    return steady_curve



