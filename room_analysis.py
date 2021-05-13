import pandas as pd
import numpy as np
import os
import plotly.figure_factory as ff
import plotly.io as pio
from plotly import graph_objects as go
import plotly.express as px


def last_step_auc_heatmap(csv_name):
    path = os.path.join(os.getcwd(), 'data', csv_name)
    df = pd.read_csv(path)

    ordered_groups = df.groupby(['agent_row', 'agent_col'])

    vals = []
    for name, group in ordered_groups:
        vals.append(group['aerosol_inhaled'].values[-1])

    row1 = vals[0:5]
    row2 = vals[5:10]
    row3 = vals[10:15]
    row4 = vals[15:20]
    row5 = vals[20:]

    heatmap = [row5, row4, row3, row2, row1]

    fig = ff.create_annotated_heatmap(heatmap)
    imagename = os.path.join(os.getcwd(), "data", csv_name+'_E3_heatmap.svg')
    pio.write_image(fig, imagename, format='svg', width=1100, height=850)


def get_rolling_average(csv_name, bin_size=100):
    rolling_average = []
    path = os.path.join(os.getcwd(), 'data', csv_name)
    df = pd.read_csv(path)

    ordered_groups = df.groupby(['iteration'])

    for name, group in ordered_groups:
        if int(name) % bin_size == 0:
            rolling_average.append(group['aerosol_inhaled'].mean())

    return rolling_average


def auc_distrs_chunk(csv_name, bin_sizes=[100, 200, 300, 400, 500, 600, 1200, 2400, 2999]):
    distrs = []
    path = os.path.join(os.getcwd(), 'data', csv_name)
    df = pd.read_csv(path)

    ordered_groups = df.groupby(['iteration'])

    i = 0
    for name, group in ordered_groups:
        if i < len(bin_sizes):
            if int(name) == bin_sizes[i]:
                print(name, i)
                distrs.append(group['aerosol_inhaled'].values)
                i += 1

    return distrs

# name1 = 'new_p11_p21_p31'
# name2 = 'new_p11_p21_p32'
# name3 = 'new_p11_p23_p31'

# x = get_rolling_average(name1+'.csv')
# y = get_rolling_average(name2+'.csv')
# # z = get_rolling_average(name3+'.csv')
# last_step_auc_heatmap(name1+'.csv')
# last_step_auc_heatmap(name2+'.csv')
# # last_step_auc_heatmap(name3+'.csv')
# t = [(i*100) + 1 for i in range(30)]

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name=name1, marker_color='blue'))
# fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=name2, marker_color='green'))
# # fig.add_trace(go.Scatter(x=t, y=z, mode='lines', name=name3, marker_color='red'))
# fig.update_layout(title='Aerosol Intake for Different Sink/Source Locations', font_family='arial', font_size=20)
# fig.update_xaxes(title='Step')
# fig.update_yaxes(title='Aerosol Intake')
# imagename = os.path.join(os.getcwd(), "data", 'new_E3_Variation.svg')
# pio.write_image(fig, imagename, format='svg', width=1100, height=850)

# x = auc_distrs_chunk(name1+'.csv')
# y = auc_distrs_chunk(name2+'.csv')
# z = auc_distrs_chunk(name3+'.csv')

# df = pd.DataFrame()

# x1 = [item for sublist in x for item in sublist]
# y1 = [item for sublist in y for item in sublist]
# z1 = [item for sublist in z for item in sublist]
# x_group = [1 for val in x1]
# y_group = [2 for val in x1]
# z_group = [3 for val in x1]
# steps = []
# for i in range(len(x)):
#     for j in range(len(x[0])):
#         steps.append(i + 1)

# bin_sizes = [100, 200, 300, 400, 500, 600, 1200, 2400, 2999]
# all_bins = []
# for i in range(len(bin_sizes)):
#     for j in range(len(x[0])):
#         all_bins.append(bin_sizes[i])

# df['val'] = x1 + y1
# df['step'] = steps + steps
# df['sim_group'] = x_group + y_group
# df['bins'] = all_bins + all_bins

# fig = px.box(df, x="bins", y="val", color="sim_group")
# fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
# fig.update_xaxes(title='Seconds From Simulation Start')
# fig.update_yaxes(title='Aerosol Intake')
# imagename = os.path.join(os.getcwd(), "data", 'new_E3_boxplot.svg')
# pio.write_image(fig, imagename, format='svg', width=1100, height=850)

control = 'new_p11_p22_p31'
sims = ['new_p11_p22_p31', 'new_p11_p21_p31', 'new_p12_p21_p31', 'new_p13_p21_p31', 'new_p11_p23_p31', 'new_p11_p21_p32']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#8c564b']

control_df = pd.read_csv(os.path.join(os.getcwd(), 'data', control+'.csv'))
control_last_step = control_df[control_df['iteration'] == 2999]
calcs = []

for sim in sims:
    path = os.path.join(os.getcwd(), 'data', sim+'.csv')
    df = pd.read_csv(path)
    last_step = df[df['iteration'] == 2999]
    calcs.append(((last_step['aerosol_inhaled'].median() - control_last_step['aerosol_inhaled'].median()) / control_last_step['aerosol_inhaled'].median()) * 100)


print(calcs)
fig = go.Figure(go.Bar(
                x=calcs,
                y=sims,
                marker_color=colors,
                orientation="h"
))
fig.show()
pio.write_image(fig, os.path.join(os.getcwd(), "data", 'new_barplot.svg'), format='svg', width=1100, height=850)