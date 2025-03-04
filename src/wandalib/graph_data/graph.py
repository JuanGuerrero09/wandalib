import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Transient
## Pipeline
### Pressure

def graph_transient_pressures(df: pd.DataFrame, title="Pipeline Pressure"):
    fig, bx = plt.subplots()
    bx.plot(df["Steady Pressure"], label="Steady Pressure", color="orange")
    bx.plot(df["Maximum Pressure"], label="Maximum Pressure", color="red", linestyle="dashdot")
    bx.plot(df["Minimum Pressure"], label="Minimum Pressure", color="blue", linestyle="dashdot")
    bx.set(xlim=(df.index[0], df.index[-1]))
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [barg]")
    plt.grid()
    plt.legend()
    plt.show()
    return [fig, bx]

### Head

def graph_transient_head(df: pd.DataFrame, profile, title="Pipeline Pressure"):
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="green")
    bx.plot(df["Steady Head"], label="Steady Head", color="orange")
    bx.plot(df["Maximum Head"], label="Maximum Head", color="red", linestyle="dashdot")
    bx.plot(df["Minimum Head"], label="Minimum Head", color="blue", linestyle="dashdot")
    bx.set(xlim=(df.index[0], df.index[-1]))
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()
    plt.show()
    return [fig, bx]

# Steady

## Pipeline
### Pressure
def graph_steady_pressure(steady_curve: pd.Series, title="Pipeline Pressure"):
    fig, bx = plt.subplots()
    bx.plot(steady_curve, label="Steady State Pressure", color="blue")
    bx.set(xlim=(steady_curve.index[0], steady_curve.index[-1]))

    # Add plot details
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [barg]")
    plt.grid()
    plt.legend()    
    # plt.show() 
    return [fig, bx]
    

### Head


def graph_steady_head(steady_curve: pd.Series, profile: pd.Series, title:str ="Pipeline Head"):
    fig, bx = plt.subplots()
    bx.plot(profile, label="Profile", color="green")
    bx.plot(steady_curve, label="Steady State Head", color="orange")
    bx.set(xlim=(profile.index[0], profile.index[-1]))

    # Add plot details
    plt.title(title)
    plt.xlabel("Distance [m]")
    plt.ylabel("Head [m]")
    plt.grid()
    plt.legend()    
    # plt.show() 
    return [fig, bx]

####

def add_air_valves(bx: plt.Axes, coordinates: list[(float, float)], color: str="red", size:int =50, label:str ="Air Valves") -> None:
    """
    Añade triángulos (marcadores) en las coordenadas especificadas en el gráfico.

    Parámetros:
        bx (matplotlib.axes.Axes): El eje del gráfico donde se añadirán los triángulos.
        coordinates (list of tuples): Lista de coordenadas (x, y) donde se colocarán los triángulos.
        color (str): Color de los triángulos (por defecto es "red").
        size (int): Tamaño de los triángulos (por defecto es 50).
    """
    i = 0
    for (x, y) in coordinates:
        if i == 0:
            bx.scatter(x, y, marker="v", color=color, s=size, zorder=5, label=label)
            i += 1
            continue
        bx.scatter(x, y, marker="v", color=color, s=size, zorder=5)  # zorder asegura que estén encima de las líneas
   