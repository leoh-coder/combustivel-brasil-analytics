from pathlib import Path

import streamlit as st
import pandas as pd

from dashboard.components.filters import filtro_combustivel, filtro_periodo, aplicar_filtros
from dashboard.components.charts import grafico_evolucao, card_metrica

st.header("📈 Visão Geral Nacional")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def carregar_dados():
    return pd.read_parquet(DATA_DIR / "agg_mensal_nacional.parquet")


try:
    df = carregar_dados()
except FileNotFoundError:
    st.error("Dados não encontrados. Execute o ETL primeiro: `python scripts/run_etl.py`")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    produto = filtro_combustivel(df, key="vg_combustivel")
with col2:
    periodo = filtro_periodo(df, key="vg_periodo")

df_filtrado = aplicar_filtros(df, produto=produto, periodo=periodo)

if not df_filtrado.empty:
    df_ordenado = df_filtrado.sort_values(["ano", "mes"])
    ultimo_preco = df_ordenado.iloc[-1]["preco_medio"]
    preco_anterior = df_ordenado.iloc[-2]["preco_medio"] if len(df_ordenado) > 1 else ultimo_preco
    variacao = ((ultimo_preco - preco_anterior) / preco_anterior) * 100

    col1, col2, col3, col4 = st.columns(4)
    card_metrica(col1, "Preço Atual", f"R$ {ultimo_preco:.2f}", f"{variacao:+.2f}%")
    card_metrica(col2, "Preço Médio", f"R$ {df_filtrado['preco_medio'].mean():.2f}")
    card_metrica(col3, "Mínimo Histórico", f"R$ {df_filtrado['preco_medio'].min():.2f}")
    card_metrica(col4, "Máximo Histórico", f"R$ {df_filtrado['preco_medio'].max():.2f}")

    st.markdown("---")

    fig = grafico_evolucao(df_filtrado, produto)
    st.plotly_chart(fig, width="stretch")

    with st.expander("📋 Ver dados agregados"):
        df_tabela = df_ordenado.iloc[::-1].head(24).copy()
        for col in ["preco_medio", "preco_mediano"]:
            if col in df_tabela.columns:
                df_tabela[col] = df_tabela[col].round(2)
        st.dataframe(
            df_tabela,
            width="stretch",
        )
else:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
