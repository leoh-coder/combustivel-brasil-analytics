from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils import construir_coluna_data


TEMPLATE = "plotly_white"


def grafico_evolucao(df: pd.DataFrame, produto: str) -> go.Figure:
    subset = df[df["produto_padronizado"] == produto].copy()
    subset["data"] = construir_coluna_data(subset)
    subset = subset.sort_values("data")

    fig = px.line(
        subset,
        x="data",
        y="preco_medio",
        title=f"Evolução do Preço Médio — {produto}",
        labels={"data": "Data", "preco_medio": "Preço Médio (R$/litro)"},
        template=TEMPLATE,
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_traces(
        hovertemplate="<b>%{x|%b %Y}</b><br>Preço: R$ %{y:.2f}<extra></extra>"
    )
    fig.update_layout(hovermode="x unified")

    fig.update_xaxes(tickformat="%b %Y")

    return fig


def grafico_estados_ranking(df: pd.DataFrame, produto: str, top_n: int = 10) -> go.Figure:
    subset = df[df["produto_padronizado"] == produto]
    ranking = (
        subset.groupby("estado_sigla")["preco_medio"]
        .mean()
        .sort_values(ascending=True)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=ranking.tail(top_n).index,
        x=ranking.tail(top_n).values,
        orientation="h",
        marker_color="#e74c3c",
        name=f"Top {top_n} mais caros",
        text=[f"R$ {v:.2f}" for v in ranking.tail(top_n).values],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Ranking de Preço por Estado — {produto}",
        xaxis_title="Preço Médio (R$/litro)",
        template=TEMPLATE,
        height=400,
    )
    return fig


def grafico_comparativo_pp(df: pd.DataFrame, produto: str) -> go.Figure:
    subset = df[df["produto_padronizado"] == produto].copy()
    subset["data"] = construir_coluna_data(subset)
    subset = subset.sort_values("data")

    fig = go.Figure()

    if "preco_pp" in subset.columns:
        fig.add_trace(go.Scatter(
            x=subset["data"], y=subset["preco_pp"],
            name="Presidente Prudente",
            line=dict(width=2.5, color="#1f77b4"),
        ))
    if "preco_sp" in subset.columns:
        fig.add_trace(go.Scatter(
            x=subset["data"], y=subset["preco_sp"],
            name="São Paulo (Estado)",
            line=dict(width=2, color="#ff7f0e", dash="dash"),
        ))
    if "preco_brasil" in subset.columns:
        fig.add_trace(go.Scatter(
            x=subset["data"], y=subset["preco_brasil"],
            name="Brasil",
            line=dict(width=2, color="#2ca02c", dash="dot"),
        ))

    fig.update_layout(
        title=f"{produto}: Presidente Prudente vs SP vs Brasil",
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template=TEMPLATE,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def grafico_previsao(df_historico: pd.DataFrame, df_previsao: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_historico["data"],
        y=df_historico["preco_medio"],
        name="Histórico",
        line=dict(width=2, color="#1f77b4"),
        hovertemplate="<b>Histórico</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_previsao["data"],
        y=df_previsao["preco_medio"],
        name="Previsão",
        line=dict(width=2.5, color="#ff7f0e", dash="dash"),
        mode="lines+markers",
        marker=dict(size=8),
        hovertemplate="<b>Previsão</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Previsão de Preço — Próximos Meses",
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template=TEMPLATE,
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def card_metrica(col, titulo: str, valor: str, delta: Optional[str] = None):
    col.metric(label=titulo, value=valor, delta=delta)
