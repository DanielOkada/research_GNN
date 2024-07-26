# movie lensから共演者ネットワークを作る

import networkx as nx
import pandas as pd
import ast


def preprocess_casts():
    # credits.csvを読み込む
    credits = pd.read_csv('data/credits.csv')

    # castデータのみを取り出す
    cast = credits[["cast", "id"]]

    # ややこしいのでidをmovie_idに変更
    cast = cast.rename(columns={"id": "movie_id"})

    # キャスト情報をリストに変換
    cast["cast"] = cast["cast"].apply(lambda x: ast.literal_eval(x))
    print(cast)

    # キャスト-映画データフレームを作る
    casts = []
    for cast_column, movie_id in cast.itertuples(index=False, name=None):
        for cast_info in cast_column:
            # ややこしいのでidはcast_idにする
            casts.append({"cast_id": cast_info["id"], "name": cast_info["name"], "movie_id": movie_id})

    cast_movie = pd.DataFrame(casts, columns=["cast_id", "name", "movie_id"])
    print(cast_movie)

    # 保存
    cast_movie.to_csv('data/cast_movie.csv', index=False)


# movies_metadata.csvを読み込む
movies = pd.read_csv('data/movies_metadata.csv')[["id", "title", "revenue"]]

# ややこしいのでidをmovie_idに変更
movies = movies.rename(columns={"id": "movie_id"})

movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors='coerce')

# revenueが大きいものだけ取り出す
movies = movies[movies["revenue"] > 10000000]

# キャストの出演情報を読み込む
casts = pd.read_csv('data/cast_movie.csv')

# データフレームを統合
df = pd.merge(casts, movies, on='movie_id', how='outer')
df = df.dropna()

# 出現回数が50回以上のキャストを抽出
freq = df['cast_id'].value_counts()
freq = freq[freq >= 25]
df = df[df['cast_id'].isin(freq.index)]

G = nx.Graph()
G.add_nodes_from(df["name"], node_shape="s", node_color="black")
G.add_nodes_from(df["title"], node_shape="s")

# エッジを追加
# ノードも自動的に追加してくれる
G.add_edges_from(df[["name", "title"]].itertuples(index=False))

print(len(G.nodes.data()), "ノード数")
print(len(G.edges.data()), "エッジ数")

# 一部グラフにする
G = nx.projected_graph(G, df["name"])

import matplotlib.pyplot as plt

# レイアウトを指定して描画
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=df["name"], node_shape="o")
# nx.draw_networkx_nodes(G, pos, nodelist=df["title"], node_shape="s", node_color="red")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)
plt.show()

# 媒介中心性を計算
betweenness = nx.betweenness_centrality(G)

names = df["name"].unique()
# 結果を出力
for node, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{node}: \t{bc:.4f}")