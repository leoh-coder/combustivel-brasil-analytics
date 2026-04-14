from pathlib import Path

import streamlit as st
import duckdb

from dashboard.components.filters import filtro_combustivel, filtro_periodo, aplicar_filtros
from dashboard.components.charts import grafico_comparativo_pp, card_metrica

st.header("📍 Presidente Prudente vs SP vs Brasil")
st.markdown(
    "Comparação do preço dos combustíveis em **Presidente Prudente** "
    "com a média do estado de **São Paulo** e a média **nacional**."
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PARQUET_PATH = DATA_DIR / "combustiveis_brasil.parquet"


@st.cache_data
def carregar_comparativo():
    with duckdb.connect() as con:
        df = con.execute(f"""
            SELECT
                ano, mes, produto_padronizado,
                AVG(CASE WHEN municipio LIKE '%PRESIDENTE PRUDENTE%'
                    THEN valor_venda END) as preco_pp,
                AVG(CASE WHEN estado_sigla = 'SP'
                    THEN valor_venda END) as preco_sp,
                AVG(valor_venda) as preco_brasil
            FROM read_parquet('{PARQUET_PATH}')
            GROUP BY ano, mes, produto_padronizado
            ORDER BY ano, mes
        """).fetchdf()
    return df


try:
    df = carregar_comparativo()
except (FileNotFoundError, duckdb.IOException) as e:
    st.error(f"Dados não encontrados: {e}. Execute o pipeline ETL primeiro.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    produto = filtro_combustivel(df, key="pp_combustivel")
with col2:
    periodo = filtro_periodo(df, key="pp_periodo")

df_filtrado = aplicar_filtros(df, produto=produto, periodo=periodo)

if df_filtrado.empty or df_filtrado["preco_pp"].isna().all():
    st.warning("Sem dados de Presidente Prudente para este filtro.")
    st.stop()

ultimo = df_filtrado.dropna(subset=["preco_pp"]).sort_values(["ano", "mes"]).iloc[-1]
diff_sp = ultimo["preco_pp"] - ultimo["preco_sp"]
diff_br = ultimo["preco_pp"] - ultimo["preco_brasil"]

col1, col2, col3 = st.columns(3)
card_metrica(col1, "Pres. Prudente", f"R$ {ultimo['preco_pp']:.2f}")
card_metrica(col2, "vs São Paulo", f"R$ {ultimo['preco_sp']:.2f}", f"{diff_sp:+.2f}")
card_metrica(col3, "vs Brasil", f"R$ {ultimo['preco_brasil']:.2f}", f"{diff_br:+.2f}")

st.markdown("---")

fig = grafico_comparativo_pp(df_filtrado, produto)
fig.update_traces(hovertemplate="%{fullData.name}<br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>")
st.plotly_chart(fig, width="stretch")
st.caption(
    "Alguns períodos podem aparecer com lacunas porque a ANP não disponibilizou "
    "observações para a combinação selecionada."
)

st.markdown("### Análise")
media_pp = df_filtrado["preco_pp"].mean()
media_sp = df_filtrado["preco_sp"].mean()
media_br = df_filtrado["preco_brasil"].mean()

if media_pp > media_sp:
    st.markdown(
        f"Em média, Presidente Prudente paga **R$ {media_pp - media_sp:.2f} a mais** "
        f"que a média do estado de São Paulo."
    )
else:
    st.markdown(
        f"Em média, Presidente Prudente paga **R$ {media_sp - media_pp:.2f} a menos** "
        f"que a média do estado de São Paulo."
    )

with st.expander("📋 Ver dados mensais"):
    st.dataframe(
        df_filtrado.dropna(subset=["preco_pp"])
        .sort_values(["ano", "mes"], ascending=False)
        .head(24)
        .round(2),
        width="stretch",
    )
