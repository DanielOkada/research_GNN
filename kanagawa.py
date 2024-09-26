import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/明治35年神奈川.xlsx', sheet_name="明治35")

# "・"を削除する(Graphvizでエラーが出る)
df['役員名'] = df['役員名'].str.replace('・', '_', regex=False)

edges_df = df[["會社名", "役員名"]].dropna()
edges = edges_df.itertuples(index=False)

G = nx.Graph()
G.add_edges_from(edges)

# 1部グラフ作成
yakuin_nodes = edges_df["役員名"].unique()
G = nx.projected_graph(G, yakuin_nodes)

nx.write_graphml(G, 'data/kanagawa_yakuin.graphml')

# 最大連結成分のみでサブグラフを作成
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)

pos = nx.spring_layout(G)  # spring_layoutは一例
# 図のサイズを設定
plt.figure(figsize=(8, 8))

# ノードを描画
nx.draw_networkx_nodes(G, pos)

# エッジを描画
nx.draw_networkx_edges(G, pos)

# ノードにラベルを付ける
nx.draw_networkx_labels(G, pos, font_family="MS Gothic")

# 表示
plt.axis('off')
plt.show()

# 媒介中心性を計算
betweenness = nx.betweenness_centrality(G)

yakuin_names = df["役員名"].unique()
# 結果を出力
for node, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    if node not in yakuin_names:
        continue

    print(f"{node}: \t{bc:.4f}")