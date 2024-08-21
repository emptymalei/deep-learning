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

import datetime
import json

# +
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

data_path = Path("data/transformer-and-forecasting-papers.json")
# -

with open(data_path, "r") as fp:
    data = json.load(fp)

data.keys()

data["metadata"]

data["items"][0]

# +
labels = {
    i["id"]: {
        "first_author_last": i.get("authorString", "").split(", ")[0].split(" ")[-1],
        "arxiv_id": i.get("arxivIds", ""),
        "year": datetime.datetime.strptime(
            i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ"
        ).year,
    }
    for i in data["items"]
}

labels

# +
coordinates = {
    i["id"]: np.array(
        [
            np.log(
                datetime.datetime.strptime(
                    i["publicationDate"], "%Y-%m-%dT%H:%M:%SZ"
                ).toordinal()
            ),
            np.log(1 + i.get("forwardEdgeCount", 0)),
        ]
    )
    for i in data["items"]
}

coordinates
# -

G = nx.Graph()

for i in data["items"]:
    G.add_node(i["id"], forwardEdgeCount=i.get("forwardEdgeCount", 0))
    for b in i.get("backwardEdges", "").split(","):
        try:
            if int(b) in data["ids"]:
                G.add_edge(i["id"], int(b))
        except ValueError as e:
            print(f"{b}: {e}")

# +
fig, ax = plt.subplots(figsize=(20, 10))

node_sizes = [30 * np.log(1 + G.nodes[node]["forwardEdgeCount"]) for node in G.nodes()]

nx.draw(G, pos=coordinates, ax=ax, edge_color="lightgray", node_size=node_sizes)
for node, (x, y) in coordinates.items():
    plt.text(
        x,
        y + 0.5,
        f'{labels[node]["first_author_last"]} {labels[node]["year"]}',
        horizontalalignment="center",
    )

# -
