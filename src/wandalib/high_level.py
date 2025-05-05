# File for high level functions for Wanda

import pywanda
import pandas as pd
from wandalib.graph_data import graph_transient_head, graph_transient_pressures, graph_steady_head, graph_steady_pressure
from wandalib.get_data import get_transient_heads, get_transient_pressures, get_head_steady, get_pressure_steady

def get_transient_results(wanda_file: pywanda.WandaModel, pipes: list[str], print_messages: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_press = get_transient_pressures(wanda_file, pipes, print_messages=print_messages)
    graph_transient_pressures(df_press)
    df_head, profile = get_transient_heads(wanda_file, pipes)
    graph_transient_head(df_head, profile)
    return df_press, df_head, profile

def get_steady_results(wanda_file: pywanda.WandaModel, pipes: list[str], print_messages: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_press = get_pressure_steady(wanda_file, pipes, show_messages=print_messages)
    graph_steady_pressure(df_press)
    df_head, profile = get_head_steady(wanda_file, pipes)
    graph_steady_head(df_head, profile)
    return df_press, df_head, profile