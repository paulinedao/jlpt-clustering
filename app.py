import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import warnings

warnings.filterwarnings("ignore")

INPUT_EMB = "data/jlpt_embeddings_umap_15d.npy"
INPUT_LABELS = "data/jlpt_cluster_labels.npy"
WORDS_USED = "data/jlpt_words_used_bert.csv"  
TRANSLATIONS = "data/translation.csv"         

X = np.load(INPUT_EMB)
labels = np.load(INPUT_LABELS)

words_df = pd.read_csv(WORDS_USED)
trans_df = pd.read_csv(TRANSLATIONS)

df_words = words_df.merge(trans_df, on="Word", how="left")
assert len(X) == len(labels) == len(df_words)

# UMAP 15D to 2D 
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X)

df_viz = pd.DataFrame({
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "cluster": labels,
    "word": df_words["Word"],
    "English": df_words["English"]
})

# Figure
fig = go.Figure()

clusters = sorted(df_viz["cluster"].unique())

for c in clusters:
    subset = df_viz[df_viz["cluster"] == c]

    if c == -1:
        color = "lightgray"
        name = "Noise"
        size = 6
        opacity = 0.4
    else:
        color = None  # let Plotly assign stable color
        name = f"Cluster {c}"
        size = 8
        opacity = 0.85

    fig.add_trace(go.Scatter(
        x=subset["x"],
        y=subset["y"],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            opacity=opacity,
            color=color
        ),
        customdata=np.stack(
            [subset["word"], subset["English"]],
            axis=-1
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "English: %{customdata[1]}<extra></extra>"
        )
    ))

fig.update_layout(
    title="UMAP + HDBSCAN Clustering (Japanese N5 Words)",
    height=700,
    legend_title_text="Cluster",
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2"
)

# Dassh app
app = Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"},
    children=[

        html.H1(
            "Semantic Clustering of Japanese N5 Vocabulary",
            style={"textAlign": "center"}
        ),

        html.P(
            "This project explores the semantic structure of basic Japanese vocabulary "
            "by applying modern language embeddings and unsupervised clustering techniques."
            "Japanese N5-level words are embedded using a Japanese Sentence-BERT model, "
            "reduced in dimensionality with UMAP, and grouped using the HDBSCAN clustering algorithm. "
            "The resulting clusters reveal semantic relationships between words without relying on "
            "English translations or predefined categories.",
            style={
                "textAlign": "justify",
                "maxWidth": "900px",
                "margin": "10px auto",
                "lineHeight": "1.6"
            }
        ),
        html.P(
            "Hover over a point to see the Japanese word and its English translation. "
            "Click a point to copy the word for dictionary lookup.",
            style={
                "textAlign": "justify",
                "maxWidth": "900px",
                "margin": "10px auto",
                "lineHeight": "1.6"
            }
        ),

        dcc.Graph(id="umap-graph", figure=fig),

        html.Div(
            style={"padding": "20px 0"},
            children=[
                html.Label("Japanese word:"),
                dcc.Input(
                    id="selected-word",
                    readOnly=True,
                    style={"width": "300px"}
                ),
                html.Br(),
                html.Label("English translation:"),
                dcc.Input(
                    id="selected-translation",
                    readOnly=True,
                    style={"width": "300px"}
                ),
            ]
        )
    ]
)


@app.callback(
    Output("selected-word", "value"),
    Output("selected-translation", "value"),
    Input("umap-graph", "clickData")
)
def on_click(clickData):
    if not clickData:
        return "", ""

    word, english = clickData["points"][0]["customdata"]
    return word, english


if __name__ == "__main__":
    app.run(debug=True)
