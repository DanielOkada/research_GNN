import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

graphml_file = 'data/kanagawa_yakuin.graphml'

# GraphMLファイルの読み込み
G = nx.read_graphml(graphml_file)

# edges_df = df[["會社名", "役員名"]].dropna()
# edges = edges_df.itertuples(index=False)
#
# G = nx.Graph()
# G.add_edges_from(edges)
#
# # 1部グラフ作成
# yakuin_nodes = edges_df["役員名"].unique()
# G = nx.projected_graph(G, yakuin_nodes)

# 可視化
plt.figure(figsize=(8,5)) # 適切なサイズで
nx.draw(G, pos=nx.nx_pydot.graphviz_layout(G), with_labels=True, font_family="MS Gothic")
plt.show()