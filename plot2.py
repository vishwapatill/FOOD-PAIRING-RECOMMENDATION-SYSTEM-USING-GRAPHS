import random
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import itertools
import operator
import datetime
import chart_studio.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
graph=pickle.load(open("D:\Food_pairing\\final\data\output\graph_all","rb"))

def plot_graph(G):
    pos = nx.spring_layout(G)  # Define the positions of nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=12)
# plot_graph(graph)

flatten = lambda l: [item for sublist in l for item in sublist]

# Prettify ingredients
pretty_food = lambda s: ' '.join(s.split('_')).capitalize().lstrip()
# Prettify cuisine names
pretty_category = lambda s: ''.join(map(lambda x: x if x.islower() else " "+x, s)).lstrip()

"""
Plot Points with Labels
"""
def make_plot_only_labels(name, points, labels, publish):
    traces = []
    traces.append(go.Scattergl(
            x = points[:, 0],
            y = points[:, 1],
            mode = 'markers',
            marker = dict(
                color = sns.xkcd_rgb["black"],
                size = 8,
                opacity = 0.6,
                #line = dict(width = 1)
            ),
            text = labels,
            hoverinfo = 'text',
        )
        )

    layout = go.Layout(
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            #autotick=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            #autotick=True,
            ticks='',
            showticklabels=False
        )
        )

    fig = go.Figure(data=traces, layout=layout)
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    plotter(fig, filename=name + '.html')

"""
Plot Points with Labels and Legends
"""

def make_plot_with_labels_legends(name, points, labels, label_to_plot, legend_labels, legend_order, legend_label_to_color, legend_label_marker, legend_label_size, pretty_legend_label, publish):
    lst = zip(points, labels, legend_labels)
    full = sorted(lst, key=lambda x: x[2])
    traces = []
    annotations = []
    
#     with open("./input/dict_ingr2count.pickle", "rb") as pickle_file:
#         ingr2count = pickle.load(pickle_file)
    
    for legend_label, group in itertools.groupby(full, lambda x: x[2]):
        group_points = []
        group_labels = []
        for tup in group:
            point, label, _ = tup
            group_points.append(point)
            group_labels.append(label)
            
        # label, legend_label
        # markers
        group_points = np.stack(group_points)
        traces.append(go.Scattergl(
            x = group_points[:, 0],
            y = group_points[:, 1],
            mode = 'markers',
            marker = dict(
                symbol = legend_label_marker[legend_label],
                color = legend_label_to_color[legend_label],
                size = legend_label_size[legend_label],
                opacity = 1,
                line = dict(width = 0.5)
            ),
            text = ['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
            #text = ['{}'.format(label) for label in group_labels],
            #textposition='middle center',
            #textfont=dict(family='sans serif', size = label_to_size[legend_label], color='black'),
            hoverinfo = 'text',
            name = legend_label
        )
        )
    
    # order the legend
    ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
    traces_ordered = flatten(ordered)
    def _set_name(trace):
        trace.name = pretty_legend_label(trace.name)
        return trace
    traces_ordered = list(map(_set_name, traces_ordered))

    layout = go.Layout(
        xaxis=dict(
            autorange=True,
            showgrid=True,
            gridcolor='silver',
            gridwidth=0.01,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=3,
            showline=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=True,
            gridcolor='silver',
            gridwidth=0.01,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=3,
            showline=True,
            ticks='',
            showticklabels=False
        ),
        annotations=annotations,
        title='FlavorNet2.0',   
        font=dict(size=12),
        showlegend=True,
        autosize=True,
        hovermode='closest',
        #paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig = go.Figure(data=traces_ordered, layout=layout)
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    plotter(fig, filename=name + '.html')

def plot_category(node2vec, node2vec_tsne, path, node2name, node2is_hub, withLegends=False):
    #Label Load
    labels = []
    for label in node2vec:
        labels.append(label)

    #Legend Load
    if withLegends:
        categories = []
        for label in labels:
            try:
                if node2is_hub[label] == 'hub':
                    categories.append('Hub_Ingredient')
                elif node2is_hub[label] == 'no_hub':
                    categories.append('Non_hub_Ingredient')
                elif node2is_hub[label] == 'food':
                    categories.append('Food_like_Compound')
                elif node2is_hub[label] == 'drug':
                    categories.append('Drug_like_Compound')
                else:
                    print(label)
            except KeyError:
                #print(label)
                categories.append("None")
        categories_color = list(set(categories))
        #print(categories_color)

        category2color = {
        'Hub_Ingredient' :  sns.xkcd_rgb["orange"],
        'Non_hub_Ingredient' : sns.xkcd_rgb["goldenrod"],
        'Food_like_Compound' : sns.xkcd_rgb["green"],
        'Drug_like_Compound'  : sns.xkcd_rgb["pink"],
        'None'  : sns.xkcd_rgb["black"]
        }
        
        category2marker = {
        'Hub_Ingredient' : 'diamond-x',
        'Non_hub_Ingredient' : 'square',
        'Food_like_Compound' : 'circle',
        'Drug_like_Compound' : 'circle'
        }
        
        category2size = {
        'Hub_Ingredient' : 14,
        'Non_hub_Ingredient' : 8,
        'Food_like_Compound' : 8,
        'Drug_like_Compound' : 9
        }
        
        label2plot = {
        'Hub_Ingredient' : 50,
        'Non_hub_Ingredient' : 200,
        'Food_like_Compound' : 100,
        'Drug_like_Compound' : 50
        }
        
        category_order = ['Non_hub_Ingredient', 'Food_like_Compound', 'Drug_like_Compound', 'Hub_Ingredient']

        make_plot_with_labels_legends(name=path,
        points=node2vec_tsne,
        labels=labels,
        label_to_plot=label2plot,
        legend_labels=categories,
        legend_order=category_order,
        legend_label_marker=category2marker,
        legend_label_size=category2size,
        legend_label_to_color=category2color,
        pretty_legend_label=pretty_category,
        publish=False)

    else:
        make_plot_only_labels(name=path,
                points=node2vec_tsne,
                labels=labels,
                publish=False)
        
def load_TSNE(ingr2vec, dim=2):
    print("\nt-SNE Started... ")
    time_start = time.time()

    X = []
    for x in ingr2vec:
        X.append(ingr2vec[x])
    X_array = np.array(X)
    tsne = TSNE(n_components=dim)
    X_tsne = tsne.fit_transform(X_array)

    print("t-SNE done!")
    print("Time elapsed: {} seconds".format(time.time()-time_start))

    return X_tsne

# Embedding

def plot_embedding():
    global graph
    print("\nPlot Embedding...")
    node2node_name={}
    node_name2is_hub={}
    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = node_info['name']
        node2node_name[node] = node_name
        node_name2is_hub[node_name] = node_info['is_hub']
    with open("D:\Food_pairing\\final\model\embedings.pickle", "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    node_name2vec = {}
    for node in vectors:
        node_name = node2node_name[int(node)]
        node_name2vec[node_name] = vectors[node]

    # TSNE
    node_name2vec_tsne = load_TSNE(node_name2vec, dim=2)

    # SAVE
    save_path = "D:\Food_pairing\\final\eval"
    plot_category(node_name2vec, node_name2vec_tsne, save_path, node2node_name, node_name2is_hub, True)

    return



# Display the graph
plt.show() 

plot_embedding()


"""
TSNE of Ingredient2Vec

"""


"""
Load functions for plotting a graph
"""

