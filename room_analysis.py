import pandas as pd
import numpy as np
import os
import plotly.figure_factory as ff
import plotly.io as pio


def last_step_auc_heatmap(csv_name):
    path = os.path.join(os.getcwd(), 'data', csv_name)
    df = pd.read_csv(path)

    ordered_groups = df.groupby(['agent_row', 'agent_col'])

    vals = []
    for name, group in ordered_groups:
        vals.append(group['agent_auc'].values[-1])

    row1 = vals[0:5]
    row2 = vals[5:10]
    row3 = vals[10:15]
    row4 = vals[15:20]
    row5 = vals[20:]

    heatmap = [row5, row4, row3, row2, row1]

    fig = ff.create_annotated_heatmap(heatmap)
    imagename = os.path.join(os.getcwd(), "data", csv_name+'_heatmap.svg')
    pio.write_image(fig, imagename, format='svg', width=1100, height=850)


def get_rolling_average(csv_name, bin_size=100):
    rolling_average = []
    path = os.path.join(os.getcwd(), 'data', csv_name)
    df = pd.read_csv(path)

    ordered_groups = df.groupby(['iteration'])

    for name, group in ordered_groups:
        if int(name) % 100 == 0:
            rolling_average.append(group['agent_auc'].mean())

    return rolling_average
