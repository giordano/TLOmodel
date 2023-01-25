import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

df = pd.read_excel('/Users/j_collins/PycharmProjects/TLOmodel/src/scripts/maternal_perinatal_analyses/'
                   'analysis_scripts/pregnancy_sankey_data.xlsx')

unique_source_target = list(pd.unique(df[['source','target']].values.ravel('k')))

mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
df['source'] = df['source'].map(mapping_dict)
df['target'] = df['target'].map(mapping_dict)
df_dict = df.to_dict(orient='list')

l = ['Pregnancy_0',
     'Uterine Pregnancy_1',
     'Successful Pregnancy_2',
     'Maternal Survival_3',
     'Ectopic Pregnancy_1',
     'Induced Abortion_2',
     'Spontaneous Abortion_2',
     'Antenatal Stillbirth_2',
     'Maternal Death_3',
     'Live Birth_4',
     'Intrapartum Stillbirth_4']

def nodify(node_names):
    node_names = l
    # uniqe name endings
    ends = sorted(list(set([e[-1] for e in node_names])))

    # intervals
    steps = 1 / len(ends)

    # x-values for each unique name ending
    # for input as node position
    nodes_x = {}
    xVal = 0
    for e in ends:
        nodes_x[str(e)] = xVal
        xVal += steps

    # x and y values in list form
    x_values = [nodes_x[n[-1]] for n in node_names]
    y_values = [0.1] * len(x_values)

    return x_values, y_values

nodified = nodify(node_names=l)

fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=unique_source_target,
        x=nodified[0],
        y=nodified[1],
        color='green'
    ),
    link=dict(
        source=df['source'],
        target=df['target'],
        value=df['value']
    )

)
])
fig.update_layout(title='Pregnancy to Birth')
fig.show()
