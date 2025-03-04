# File for high level functions for Wanda

import pywanda
from wandalib import *

def get_transient_results(wanda_file: pywanda.WandaModel, pipes: list[str]):
    df_press = get_transient_pressures(wanda_file, pipes)
    graph_transient_pressures(df_press)
    df_head, profile = get_transient_heads(wanda_file, pipes)
    graph_transient_head(df_head, profile)
    return df_press, df_head, profile