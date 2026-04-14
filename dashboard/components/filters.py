from typing import Optional

import streamlit as st
import pandas as pd


def filtro_combustivel(df: pd.DataFrame, key: str = "combustivel") -> str:
    produtos = sorted(
        p for p in df["produto_padronizado"].dropna().unique()
        if p != "Diesel S50"
    )
    if not produtos:
        st.warning("Nenhum combustível disponível para seleção.")
        st.stop()
    default = produtos.index("Gasolina Comum") if "Gasolina Comum" in produtos else 0
    return st.selectbox("Combustível", produtos, index=default, key=key)


def filtro_estado(df: pd.DataFrame, key: str = "estado", multi: bool = False):
    estados = sorted(df["estado_sigla"].unique())
    if multi:
        return st.multiselect("Estados", estados, default=estados[:5], key=key)
    return st.selectbox("Estado", estados, key=key)


def filtro_periodo(df: pd.DataFrame, key: str = "periodo") -> tuple:
    ano_min = int(df["ano"].min())
    ano_max = int(df["ano"].max())
    return st.slider(
        "Período",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max),
        key=key,
    )


def aplicar_filtros(
    df: pd.DataFrame,
    produto: Optional[str] = None,
    estados: Optional[list] = None,
    periodo: Optional[tuple] = None,
    bandeiras: Optional[list] = None,
) -> pd.DataFrame:
    if produto:
        df = df[df["produto_padronizado"] == produto]
    if estados:
        df = df[df["estado_sigla"].isin(estados)]
    if periodo:
        df = df[(df["ano"] >= periodo[0]) & (df["ano"] <= periodo[1])]
    if bandeiras:
        df = df[df["bandeira"].isin(bandeiras)]
    return df
