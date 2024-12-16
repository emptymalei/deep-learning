# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # History of Transformer for Forecasting

# **This is a temporary notebook to produce figures for my book.**

import datetime
import json

# +
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

data_path = Path("data/transformer-and-forecasting-papers.json")
# -

with open(data_path, "r") as fp:
    data = json.load(fp)

tags = (
    pd.read_csv("data/transformer-and-forecasting-papers.csv")
    .set_index("LitmapsId")
    .Tags.to_dict()
)

# +
# labels = {
#     i["id"]: {
#         "first_author_last": i.get("authorString", "").split(", ")[0].split(" ")[-1],
#         "arxiv_id": i.get("arxivIds", ""),
#         "year": datetime.datetime.strptime(i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ").year
#     }
#     for i in
#     data["items"]
# }
# -

data["items"][0]

min_date = min(
    [
        datetime.datetime.strptime(
            i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ"
        ).toordinal()
        for i in data["items"]
    ]
)

# +
rng = np.random.default_rng(42)

coordinate_calculator = lambda i: np.array(
    [
        np.log(
            datetime.datetime.strptime(
                i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ"
            ).toordinal()
        ),
        np.log(1 + i.get("forwardEdgeCount", 0)) * (1 + rng.random() * 0.2)
        if i.get("forwardEdgeCount", 0) > 50
        else np.log(1 + i.get("forwardEdgeCount", 0)) + rng.random() * 3,
    ]
)

coordinates = {i["id"]: coordinate_calculator(i) for i in data["items"]}

color_calculator = lambda i: "black" if "Foundation Model" in tags[i["id"]] else "red"


marker_calculator = lambda i: "s" if "Foundation Model" in tags[i["id"]] else "o"

node_size_calculator = lambda x: 10 + 30 * np.log(1 + x)

# +
G = nx.Graph()

for i in data["items"]:
    G.add_node(
        i["id"],
        forwardEdgeCount=i.get("forwardEdgeCount", 0),
        coordinate=coordinates[i["id"]],
        color=color_calculator(i),
        marker=marker_calculator(i),
        size=node_size_calculator(i.get("forwardEdgeCount", 0)),
        first_author_last=i.get("authorString", "").split(", ")[0].split(" ")[-1],
        arxiv_id=i.get("arxivIds", ""),
        year=datetime.datetime.strptime(
            i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ"
        ).year,
        title=i["title"],
    )
    for b in i.get("backwardEdges", "").split(","):
        try:
            if int(b) in data["ids"]:
                G.add_edge(i["id"], int(b))
        except ValueError as e:
            print(f"{b}: {e}")

# +
transformer_origin_id = 261105339

# title_labels = {
#     i["id"]: i["title"]
#     for i in
#     data["items"]
#     if i.get("forwardEdgeCount",0) >= 1000
# }


# +
fig, ax = plt.subplots(figsize=(25, 25))


node_size_calculator = lambda x: 10 + 30 * np.log(1 + x)

nx.draw_networkx_edges(G, coordinates, edge_color="lightgray")

# Draw the nodes with different markers
effective_nodes = []
for node in G.nodes:
    try:
        nx.draw_networkx_nodes(
            G,
            pos=coordinates,
            nodelist=[node],
            node_size=G.nodes.get(node).get("size"),
            node_shape=G.nodes.get(node).get("marker"),
            node_color=G.nodes.get(node).get("color"),
        )
        effective_nodes.append(node)
    except IndexError as e:
        print(f"{node}: {e}")

for node in effective_nodes:
    x, y = G.nodes.get(node).get("coordinate")
    first_author_last = G.nodes.get(node).get("first_author_last")
    year = G.nodes.get(node).get("year")
    plt.text(x, y + 0.1, f"{first_author_last} {year}", horizontalalignment="center")

for node in effective_nodes:
    node_citations = G.nodes.get(node).get("forwardEdgeCount")
    if node_citations >= 1000:
        node_title = G.nodes.get(node).get("title")
        plt.text(
            coordinates[node][0],
            coordinates[node][1] - 0.3,
            node_title,
            horizontalalignment="center",
        )

plt.axis("off")
fig.savefig(
    "results/transformer_history/transformer_ts_papers.png",
    bbox_inches="tight",
    pad_inches=-0.5,
    # transparent=True
)
# -
# ## Table

# +
df = pd.DataFrame(data["items"]).rename(
    columns={
        "forwardEdgeCount": "citations",
        "authorString": "authors",
    }
)

df["year"] = df.publicationDate.apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").year
)
df.fillna({"citations": 0}, inplace=True)
df = df.astype(dtype={"citations": int})[["title", "authors", "citations", "year"]]

df.sort_values(by="year", inplace=True)
# -


print(df.to_markdown())
