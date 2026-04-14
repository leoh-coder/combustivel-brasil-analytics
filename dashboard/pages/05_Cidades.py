from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PARQUET_PATH = DATA_DIR / "combustiveis_brasil.parquet"

PRODUTOS = ["Gasolina Comum", "Etanol", "Diesel", "Diesel S10", "GNV"]

st.header("🏙️ Comparativo de Cidades e Estados")
st.markdown(
    "Digite o nome de uma cidade ou estado no campo abaixo e selecione o que quiser comparar. "
    "**Misture livremente** — ex: *Presidente Prudente vs Londrina vs Estado da Bahia*."
)


# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_data
def montar_opcoes() -> list:
    with duckdb.connect() as con:
        df_cid = con.execute(f"""
            SELECT municipio, estado_sigla
            FROM read_parquet('{PARQUET_PATH}')
            WHERE municipio IS NOT NULL AND LENGTH(TRIM(municipio)) > 1
            GROUP BY municipio, estado_sigla
            HAVING COUNT(*) >= 50
            ORDER BY municipio
        """).fetchdf()

        df_uf = con.execute(f"""
            SELECT DISTINCT estado_sigla
            FROM read_parquet('{PARQUET_PATH}')
            WHERE estado_sigla IS NOT NULL
            ORDER BY estado_sigla
        """).fetchdf()

    cidades = [
        f"🏙 {row['municipio'].title()} ({row['estado_sigla']})"
        for _, row in df_cid.iterrows()
    ]
    estados = [f"🗺 Estado: {uf}" for uf in df_uf["estado_sigla"]]
    return cidades + estados + ["🌎 Média Nacional (Brasil)"]


@st.cache_data
def serie_cidade(municipio: str, produto: str) -> pd.DataFrame:
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT ano, mes, AVG(valor_venda) AS preco_medio
            FROM read_parquet('{PARQUET_PATH}')
            WHERE municipio = ? AND produto_padronizado = ?
            GROUP BY ano, mes ORDER BY ano, mes
        """, [municipio, produto]).fetchdf()


@st.cache_data
def serie_estado(estado: str, produto: str) -> pd.DataFrame:
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT ano, mes, AVG(valor_venda) AS preco_medio
            FROM read_parquet('{PARQUET_PATH}')
            WHERE estado_sigla = ? AND produto_padronizado = ?
            GROUP BY ano, mes ORDER BY ano, mes
        """, [estado, produto]).fetchdf()


@st.cache_data
def serie_brasil(produto: str) -> pd.DataFrame:
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT ano, mes, AVG(valor_venda) AS preco_medio
            FROM read_parquet('{PARQUET_PATH}')
            WHERE produto_padronizado = ?
            GROUP BY ano, mes ORDER BY ano, mes
        """, [produto]).fetchdf()


def _para_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["data"] = pd.to_datetime(
        df[["ano", "mes"]].rename(columns={"ano": "year", "mes": "month"}).assign(day=1)
    )
    return df


try:
    todas_opcoes = montar_opcoes()
except (FileNotFoundError, duckdb.IOException) as e:
    st.error(f"Dados não encontrados: {e}. Execute o pipeline ETL primeiro.")
    st.stop()

col_prod, col_sel = st.columns([1, 3])

with col_prod:
    produto = st.selectbox("Combustível", PRODUTOS, key="cid_produto")

with col_sel:
    selecionados = st.multiselect(
        "🔍 Buscar e selecionar (cidades, estados ou Brasil — máx. 6)",
        todas_opcoes,
        max_selections=6,
        key="cid_multi",
        placeholder="Digite 'Prudente', 'Londrina', 'BA'...",
    )

if not selecionados:
    st.info(
        "💡 **Como usar:** no campo acima, comece a digitar o nome de uma cidade "
        "(ex: *Prud*) ou sigla de estado (ex: *BA*) e selecione o que aparecer. "
        "Você pode misturar cidades e estados à vontade."
    )
    st.stop()

OPCAO_BRASIL = "🌎 Média Nacional (Brasil)"
series: dict[str, pd.DataFrame] = {}
tipos: dict[str, str] = {}

for sel in selecionados:
    if sel == OPCAO_BRASIL:
        series[sel] = serie_brasil(produto)
        tipos[sel] = "brasil"

    elif sel.startswith("🗺 Estado: "):
        uf = sel.replace("🗺 Estado: ", "")
        series[sel] = serie_estado(uf, produto)
        tipos[sel] = "estado"

    elif sel.startswith("🏙 "):
        parte = sel.replace("🏙 ", "")
        municipio_title, uf = parte.rsplit(" (", 1)
        municipio = municipio_title.upper()
        series[sel] = serie_cidade(municipio, produto)
        tipos[sel] = "cidade"

anos_validos = [df["ano"].values for df in series.values() if not df.empty]
if not anos_validos:
    st.warning("Sem dados para as seleções e produto escolhidos.")
    st.stop()

ano_min = int(min(a.min() for a in anos_validos))
ano_max = int(max(a.max() for a in anos_validos))
periodo = st.slider("Período", ano_min, ano_max, (ano_min, ano_max), key="cid_periodo")

series_base: dict[str, pd.DataFrame] = {}
for label, df in series.items():
    if df.empty:
        continue
    df_f = df[(df["ano"] >= periodo[0]) & (df["ano"] <= periodo[1])]
    if not df_f.empty:
        serie = _para_data(df_f)[["data", "preco_medio"]].sort_values("data")
        series_base[label] = serie

if not series_base:
    st.warning("Sem dados no período selecionado.")
    st.stop()

series_f: dict[str, pd.DataFrame] = {}
for label, serie in series_base.items():
    idx_periodo = pd.date_range(
        start=serie["data"].min(),
        end=serie["data"].max(),
        freq="MS",
    )
    series_f[label] = (
        serie
            .set_index("data")
            .reindex(idx_periodo)
            .rename_axis("data")
            .reset_index()
    )

st.markdown("### Último preço registrado")

ref_brasil = None
if OPCAO_BRASIL in series_f and not series_f[OPCAO_BRASIL].empty:
    brasil_valido = series_f[OPCAO_BRASIL].dropna(subset=["preco_medio"])
    if not brasil_valido.empty:
        ref_brasil = float(brasil_valido.iloc[-1]["preco_medio"])

cols = st.columns(len(series_f))
for i, (label, df_s) in enumerate(series_f.items()):
    validos = df_s.dropna(subset=["preco_medio"])
    if validos.empty:
        cols[i].metric(label=label, value="Sem dado")
        continue
    ultimo = float(validos.iloc[-1]["preco_medio"])
    delta = None
    if ref_brasil is not None and label != OPCAO_BRASIL:
        delta = f"{ultimo - ref_brasil:+.2f} vs Brasil"
    cols[i].metric(label=label, value=f"R$ {ultimo:.2f}", delta=delta)

st.markdown("---")

CORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig = go.Figure()
for idx, (label, df_s) in enumerate(series_f.items()):
    tipo = tipos[label]
    dash = "solid" if tipo == "cidade" else "dash"
    largura = 2 if tipo == "cidade" else 2.5
    cor = CORES[idx % len(CORES)]

    fig.add_trace(go.Scatter(
        x=df_s["data"],
        y=df_s["preco_medio"].round(2),
        mode="lines",
        name=label,
        connectgaps=False,
        line=dict(color=cor, dash=dash, width=largura),
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: R$ %{{y:.2f}}<extra></extra>",
    ))

fig.update_layout(
    title=f"Evolução do preço — {produto}",
    xaxis_title="Período",
    yaxis_title="Preço médio (R$/L)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=480,
    template="plotly_white",
)

st.plotly_chart(fig, width="stretch")
st.caption("Linhas sólidas = cidades · tracejadas = médias estaduais / nacional.")

with st.expander("📋 Ver dados mensais"):
    frames = [
        df_s[["data", "preco_medio"]].assign(serie=label)
        for label, df_s in series_f.items()
    ]
    df_long = pd.concat(frames)
    pivot = (
        df_long.pivot_table(index="data", columns="serie", values="preco_medio")
        .sort_index(ascending=False)
        .head(48)
        .round(2)
    )
    pivot.index = pivot.index.strftime("%b %Y")
    for _en, _pt in [("Feb","Fev"),("Apr","Abr"),("May","Mai"),
                     ("Aug","Ago"),("Sep","Set"),("Oct","Out"),("Dec","Dez")]:
        pivot.index = pivot.index.str.replace(_en, _pt)
    st.dataframe(pivot, width="stretch")

st.markdown("---")
st.caption(
    "**Fonte:** ANP — Agência Nacional do Petróleo  ·  "
    "Cidades com ≥ 50 registros  ·  "
    "Linhas sólidas = cidades · Tracejadas = médias estaduais / nacional  ·  "
    "Lacunas podem ocorrer quando não há observação da ANP no período."
)
