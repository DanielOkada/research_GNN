import glob
import random
from preprocess import *
from torch_geometric.loader import DataLoader

labels = {"circle": 0, "square": 1, "star": 2, "triangle": 3}

dataset = []
for label in labels:
    image_paths = glob.glob(f"shapes/{label}/*")
    random.shuffle(image_paths)
    for path in image_paths[:800]:
        data = create_graph_from_contour(path, labels[label])
        dataset.append(data)

import networkx as nx

graphs = []
labels = []
for data in dataset:
    # Create Networkx.classes
    e_list = []
    tensor_edgelist = data.edge_index
    for i in range(len(tensor_edgelist[0])):
        e_list.append((int(tensor_edgelist[0][i]), int(tensor_edgelist[1][i])))
    g = nx.from_edgelist(e_list)

    # Load features
    x = data.x
    #nx.set_node_attributes(g, {j: x[j] for j in range(g.number_of_nodes())}, "feature")
    nx.set_node_attributes(g, {j: str(j) for j in range(g.number_of_nodes())}, "feature")

    # Checking the consecutive numeric indexing.
    node_indices = sorted([node for node in g.nodes()])
    numeric_indices = [index for index in range(g.number_of_nodes())]

    if numeric_indices == node_indices:
        graphs.append(g)
        labels.append(int(data.y))
    else:
        pass


import seaborn as sns
import matplotlib.pyplot as plt
from karateclub.graph_embedding import Graph2Vec

model = Graph2Vec(wl_iterations=2, use_node_attribute="feature", dimensions=256,
                  down_sampling=0.0001, epochs=100, learning_rate=0.025, min_count=10)
model.fit(graphs)
emb = model.get_embedding() # (1108, 128)

sns.clustermap(emb)
plt.show()