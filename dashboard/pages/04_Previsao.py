from pathlib import Path
from io import StringIO

import joblib
import pandas as pd
import streamlit as st

from dashboard.components.charts import card_metrica, grafico_previsao
from dashboard.components.filters import filtro_combustivel
from src.eda import plot_contrafactual, plot_holt_winters_vs_real, plot_previsto_vs_real
from src.ml import (
    analise_contrafactual_holt_winters,
    comparar_modelos,
    preparar_features_temporais,
    prever_holt_winters,
    prever_proximo_periodo,
    treinar_holt_winters,
    treinar_modelo_previsao,
)
from src.utils import construir_coluna_data


st.header("🔮 Previsão de Preço")
st.markdown(
    "Compare modelos, valide o desempenho recente e projete o preço médio "
    "com base no histórico mensal da ANP."
)

st.info(
    "📐 **Escalas diferentes:** Holt-Winters trabalha em **R$/litro**. "
    "Random Forest e Gradient Boosting trabalham com **variação mensal (%)** "
    "e depois reconstroem o preço. Compare métricas apenas dentro da mesma escala."
)

with st.expander("ℹ️ Como ler esta página"):
    st.markdown(
        """
- **Comparativo de Modelos**: benchmark técnico dos modelos na validação temporal.
- **Previsto vs Real**: desempenho recente do modelo selecionado fora da amostra.
- **Projeção dos Próximos Meses**: estimativa futura para o produto escolhido.
- **Contrafactual**: cenário hipotético sem os choques de 2020+, treinado até 2019.

**Resumo:** cada bloco responde uma pergunta diferente.  
Não compare *Previsto vs Real* e *Contrafactual* como se fossem a mesma análise.
        """
    )

st.markdown("---")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


@st.cache_data
def carregar_dados():
    return pd.read_parquet(DATA_DIR / "agg_mensal_nacional.parquet")


@st.cache_data(show_spinner="Calculando análise contrafactual...")
def carregar_contrafactual(produto: str) -> dict:
    df = pd.read_parquet(DATA_DIR / "agg_mensal_nacional.parquet")
    df_produto = df[df["produto_padronizado"] == produto].copy()
    return analise_contrafactual_holt_winters(df_produto, ano_corte=2019)


@st.cache_data(show_spinner="Comparando modelos...")
def calcular_comparacao(df_produto_json: str) -> pd.DataFrame:
    df = pd.read_json(StringIO(df_produto_json), orient="split")
    df_feat = preparar_features_temporais(
        df,
        coluna_preco="preco_medio",
        target_retorno=True,
    )
    return comparar_modelos(
        df_feat,
        coluna_preco="preco_medio",
        n_splits=5,
        target_retorno=True,
        df_original=df,
    )


try:
    df_mensal = carregar_dados()
except FileNotFoundError:
    st.error("Dados não encontrados. Execute o pipeline ETL primeiro.")
    st.stop()


col1, col2, col3 = st.columns(3)

with col1:
    produto = filtro_combustivel(df_mensal, key="prev_combustivel")

with col2:
    modelo_tipo = st.selectbox(
        "Modelo",
        ["holt_winters", "random_forest", "gradient_boosting"],
        format_func=lambda x: {
            "holt_winters": "Holt-Winters (Exp. Smoothing)",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
        }[x],
        key="prev_modelo",
    )

with col3:
    meses_futuro = st.slider("Meses de previsão", 1, 12, 6, key="prev_meses")


if modelo_tipo == "holt_winters":
    st.caption("✅ Holt-Winters é a referência principal para projeção futura em preço.")
else:
    st.caption(
        "ℹ️ Modelo em variação mensal. Para leitura principal em R$/litro, "
        "use Holt-Winters como referência."
    )


df_produto = (
    df_mensal[df_mensal["produto_padronizado"] == produto]
    .sort_values(["ano", "mes"])
    .copy()
    .reset_index(drop=True)
)

if len(df_produto) < 24:
    st.warning("Dados insuficientes para treinar o modelo (mínimo: 24 meses).")
    st.stop()


modelo_path = MODELS_DIR / f"{produto.replace(' ', '_')}_{modelo_tipo}.joblib"
resultado = None
df_features = None


if modelo_tipo == "holt_winters":
    if modelo_path.exists():
        resultado = joblib.load(modelo_path)
        modelo_obj = resultado.get("modelo")
        is_damped = getattr(getattr(modelo_obj, "model", None), "damped_trend", False)

        if resultado.get("tipo") != "holt_winters" or not is_damped:
            modelo_path.unlink()
            resultado = None

    if resultado is None:
        with st.spinner("Treinando Holt-Winters pela primeira vez (será salvo para uso futuro)..."):
            resultado = treinar_holt_winters(df_produto, coluna_preco="preco_medio")
            joblib.dump(resultado, modelo_path)
        st.caption("✅ Modelo treinado e salvo em disco.")
    else:
        st.caption(
            "✅ Modelo carregado do disco — treinado em "
            f"{pd.Timestamp(modelo_path.stat().st_mtime, unit='s').strftime('%d/%m/%Y %H:%M')}"
        )

    previsoes = prever_holt_winters(resultado, meses_futuro=meses_futuro)

else:
    df_features = preparar_features_temporais(
        df_produto,
        coluna_preco="preco_medio",
        target_retorno=True,
    )

    if modelo_path.exists():
        resultado = joblib.load(modelo_path)

        if not resultado.get("target_retorno", False):
            modelo_path.unlink()
            resultado = None

    if resultado is None:
        with st.spinner("Treinando modelo pela primeira vez (será salvo para uso futuro)..."):
            resultado = treinar_modelo_previsao(
                df_features,
                coluna_preco="preco_medio",
                modelo_tipo=modelo_tipo,
                target_retorno=True,
            )
            joblib.dump(resultado, modelo_path)
        st.caption("✅ Modelo treinado e salvo em disco.")
    else:
        st.caption(
            "✅ Modelo carregado do disco — treinado em "
            f"{pd.Timestamp(modelo_path.stat().st_mtime, unit='s').strftime('%d/%m/%Y %H:%M')}"
        )

    previsoes = prever_proximo_periodo(
        resultado,
        df_features,
        meses_futuro=meses_futuro,
        target_retorno=True,
    )


st.markdown("---")
st.subheader("1) Comparativo de Modelos")
st.caption(
    "Benchmark técnico com validação temporal. Este bloco mostra desempenho dos modelos, "
    "não define sozinho qual cenário futuro deve ser lido como principal."
)

df_cmp = calcular_comparacao(df_produto.to_json(orient="split", date_format="iso"))

df_ml = df_cmp[df_cmp["Escala"] == "Variação %"]
if not df_ml.empty:
    st.warning(
        "⚠️ Em modelos de variação mensal, algumas métricas podem favorecer modelos lineares "
        "por multicolinearidade entre lags e médias móveis. Use RF/GB como benchmark complementar."
    )

df_hw_row = df_cmp[df_cmp["Modelo"] == "Holt-Winters"]
if not df_hw_row.empty:
    hw_mae = df_hw_row["MAE (média)"].iloc[0]
    hw_r2 = df_hw_row["R² (média)"].iloc[0]
    hw_r2_nota = (
        " R² negativo pode ocorrer em séries com quebras estruturais; prefira o MAE na leitura principal."
        if hw_r2 < 0
        else ""
    )

    st.info(
        f"🌊 **Holt-Winters** é a referência principal para projeção em preço. "
        f"Na validação, teve MAE de **R\\$ {hw_mae:.2f}/litro** e R² de **{hw_r2:.4f}**."
        f"{hw_r2_nota}"
    )

st.dataframe(df_cmp, width="stretch", hide_index=True)


st.markdown("---")
st.subheader("2) Validação Recente — Previsto vs Real")
st.caption(
    "Mostra como o modelo selecionado se saiu no teste mais recente (*out-of-sample*). "
    "Não é a mesma pergunta do contrafactual."
)

if modelo_tipo == "holt_winters":
    fig_pvr = plot_holt_winters_vs_real(resultado, df_produto, coluna_preco="preco_medio")
else:
    fig_pvr = plot_previsto_vs_real(resultado, df_features, coluna_preco="preco_medio")

st.plotly_chart(fig_pvr, width="stretch")


st.markdown("---")
st.subheader("Métricas do Modelo Selecionado")
st.caption(
    "Validação cruzada temporal com 5 janelas. Interprete MAE/RMSE junto com a escala do modelo."
)

_nomes_modelo = {
    "holt_winters": "Holt-Winters",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

_linha_cmp = df_cmp[df_cmp["Modelo"] == _nomes_modelo[modelo_tipo]]

if not _linha_cmp.empty:
    mae_med = float(_linha_cmp["MAE (média)"].values[0])
    mae_std = float(_linha_cmp["MAE (std)"].values[0])
    rmse_med = float(_linha_cmp["RMSE (média)"].values[0])
    rmse_std = float(_linha_cmp["RMSE (std)"].values[0])
    r2_med = float(_linha_cmp["R² (média)"].values[0])
    r2_std = float(_linha_cmp["R² (std)"].values[0])
else:
    metricas = resultado["metricas_cv"]
    mae_med, mae_std = metricas["mae"]["media"], metricas["mae"]["std"]
    rmse_med, rmse_std = metricas["rmse"]["media"], metricas["rmse"]["std"]
    r2_med, r2_std = metricas["r2"]["media"], metricas["r2"]["std"]


col1, col2, col3 = st.columns(3)

if modelo_tipo == "holt_winters":
    card_metrica(col1, "MAE", f"R$ {mae_med:.2f}")
    col1.caption(f"Erro médio em R$/litro · DP: ±{mae_std:.2f}")

    card_metrica(col2, "RMSE", f"R$ {rmse_med:.2f}")
    col2.caption(f"Penaliza erros maiores · DP: ±{rmse_std:.2f}")

    card_metrica(col3, "R²", f"{r2_med:.4f}")
    col3.caption(f"Variância explicada · DP: ±{r2_std:.4f}")
    col3.caption("Em séries com quebras estruturais, use o MAE como leitura principal.")

    pct_erro = (mae_med / 6.0) * 100
    st.info(
        f"Leitura prática: MAE de **R\\$ {mae_med:.2f}** equivale a erro médio próximo de "
        f"**{pct_erro:.0f}%** para um preço de referência de **R\\$ 6,00**."
    )

else:
    card_metrica(col1, "MAE", f"{mae_med:.4f}")
    col1.caption(f"Erro médio em variação mensal · DP: ±{mae_std:.4f}")

    card_metrica(col2, "RMSE", f"{rmse_med:.4f}")
    col2.caption(f"Penaliza erros maiores · DP: ±{rmse_std:.4f}")

    card_metrica(col3, "R²", f"{r2_med:.4f}")
    col3.caption(f"Variância explicada · DP: ±{r2_std:.4f}")


st.markdown("---")

if modelo_tipo == "holt_winters":
    df_historico = df_produto.copy()
    df_historico["data"] = construir_coluna_data(df_historico)
    df_historico = df_historico.sort_values("data").reset_index(drop=True)
else:
    df_historico = df_features.copy()
    df_historico["data"] = construir_coluna_data(df_historico)


st.subheader("3) Projeção dos Próximos Meses")

if modelo_tipo == "holt_winters":
    st.caption("Leitura principal de projeção futura em preço absoluto (R$/litro).")
else:
    st.caption(
        "Projeção derivada do modelo selecionado. Para leitura principal em preço absoluto, "
        "a referência recomendada é Holt-Winters."
    )

fig = grafico_previsao(df_historico, previsoes)
st.plotly_chart(fig, width="stretch")


st.subheader("Previsões Detalhadas")

tabela_prev = previsoes[["data", "preco_medio", "variacao_prevista"]].copy()
tabela_prev.columns = ["Data", "Preço Previsto (R$)", "Variação (%)"]

_MESES_PT = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}

tabela_prev["Data"] = tabela_prev["Data"].apply(lambda d: f"{_MESES_PT[d.month]} {d.year}")
tabela_prev["Preço Previsto (R$)"] = tabela_prev["Preço Previsto (R$)"].round(2)
tabela_prev["Variação (%)"] = (tabela_prev["Variação (%)"] * 100).round(2).astype(str) + "%"

st.dataframe(tabela_prev, width="stretch", hide_index=True)


if modelo_tipo != "holt_winters":
    with st.expander("📊 Importância das Features"):
        st.dataframe(resultado["importancia_features"], width="stretch", hide_index=True)


st.markdown("---")
st.subheader("4) Contrafactual — Treino até 2019")
st.caption(
    "Cenário hipotético: como seria a trajetória do preço sem os choques de 2020+. "
    "É análise de cenário, não métrica de acurácia."
)

try:
    resultado_cf = carregar_contrafactual(produto)
    fig_cf = plot_contrafactual(resultado_cf["df_contrafactual"], ano_corte=2019)
    st.plotly_chart(fig_cf, width="stretch")

    resumo_cf = resultado_cf["resumo"]
    col1, col2, col3 = st.columns(3)

    card_metrica(
        col1,
        "Maior impacto",
        f"R$ {resumo_cf['max_diferenca_absoluta']:.2f}",
        resumo_cf["mes_maior_impacto"],
    )
    card_metrica(col2, "Diferença média", f"R$ {resumo_cf['media_diferenca']:.2f}")
    card_metrica(col3, "Diferença máxima %", f"{resumo_cf['max_diferenca_pct']:.1f}%")

    st.caption(
        "Leitura correta: diferença entre o preço real observado e a trajetória esperada por "
        "tendência e sazonalidade pré-choques. Fatores posteriores a 2019 não entram neste cenário."
    )

except ValueError as e:
    st.info(f"Análise contrafactual indisponível para {produto}: {e}")


st.markdown("---")
st.caption(
    "**Metodologia:** CRISP-DM · **Validação:** TimeSeriesSplit (5 janelas) · "
    "**Holt-Winters:** trend=add, seasonal=add, seasonal_periods=12"
)
st.caption(
    "**RF/GB:** lags (1–6), médias móveis (3m/6m), sazonalidade, "
    "câmbio USD/BRL e preço do Brent"
)
