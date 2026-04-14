from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils import construir_coluna_data
from dashboard.components.filters import (
    filtro_combustivel,
    filtro_periodo,
    filtro_estado,
    aplicar_filtros,
)
from dashboard.components.charts import grafico_estados_ranking

st.header("🗺️ Comparativo entre Estados")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def carregar_dados():
    return pd.read_parquet(DATA_DIR / "agg_mensal_estado_produto.parquet")


try:
    df = carregar_dados()
except FileNotFoundError:
    st.error("Dados não encontrados. Execute o ETL primeiro.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    produto = filtro_combustivel(df, key="comp_combustivel")
with col2:
    periodo = filtro_periodo(df, key="comp_periodo")

df_filtrado = aplicar_filtros(df, produto=produto, periodo=periodo)

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado.")
    st.stop()

st.subheader("Ranking de Preço por Estado")
fig_ranking = grafico_estados_ranking(df_filtrado, produto, top_n=15)
st.plotly_chart(fig_ranking, width="stretch")

st.markdown("---")

st.subheader("Evolução Temporal por Estado")
estados_selecionados = filtro_estado(df_filtrado, key="comp_estados", multi=True)

if estados_selecionados:
    df_estados = df_filtrado[df_filtrado["estado_sigla"].isin(estados_selecionados)].copy()
    df_estados["data"] = construir_coluna_data(df_estados)
    df_estados = df_estados.sort_values(["estado_sigla", "data"]).reset_index(drop=True)

    fig = px.line(
        df_estados,
        x="data",
        y="preco_medio",
        color="estado_sigla",
        title=f"Evolução do Preço — {produto} por Estado",
        labels={
            "data": "Data",
            "preco_medio": "Preço Médio (R$/litro)",
            "estado_sigla": "Estado",
        },
        template="plotly_white",
    )
    fig.update_layout(hovermode="x unified")
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>")
    st.plotly_chart(fig, width="stretch")

with st.expander("📋 Tabela comparativa por estado"):
    resumo = (
        df_filtrado.groupby("estado_sigla")["preco_medio"]
        .agg(["mean", "min", "max", "std", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
        .rename(columns={
            "mean": "Preço Médio",
            "min": "Mínimo",
            "max": "Máximo",
            "std": "Desvio Padrão",
            "count": "Registros",
        })
    )
    st.dataframe(resumo, width="stretch")
