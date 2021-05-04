import pandas as pd
import numpy as np
import os
import plotly.figure_factory as ff


path = os.path.join(os.getcwd(), 'data', 'test_1.csv')
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
fig.show()