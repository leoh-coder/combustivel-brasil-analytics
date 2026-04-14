import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.utils import setup_logger, construir_coluna_data

logger = setup_logger("eda")

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100


def plot_evolucao_preco(
    df: pd.DataFrame,
    produtos = None,
    titulo: str = "Evolução do Preço Médio dos Combustíveis no Brasil",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 7))

    if produtos:
        df = df[df["produto_padronizado"].isin(produtos)].copy()

    df["data"] = construir_coluna_data(df)

    for produto in df["produto_padronizado"].unique():
        subset = df[df["produto_padronizado"] == produto].sort_values("data")
        ax.plot(subset["data"], subset["preco_medio"], label=produto, linewidth=2)

    ax.set_title(titulo, fontsize=16, fontweight="bold")
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Preço Médio (R$/litro)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("R$ %.2f"))

    eventos = {
        "2020-03": "COVID-19",
        "2022-02": "Guerra Ucrânia",
        "2023-06": "Nova política\nPetrobras",
    }
    for data_str, label in eventos.items():
        data_evento = pd.to_datetime(data_str + "-01")
        if df["data"].min() <= data_evento <= df["data"].max():
            ax.axvline(data_evento, color="gray", linestyle="--", alpha=0.5)
            ax.text(
                data_evento, ax.get_ylim()[1] * 0.95, label,
                fontsize=8, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

    plt.tight_layout()
    return fig


def plot_comparativo_estados(
    df: pd.DataFrame,
    produto: str = "Gasolina Comum",
    top_n: int = 10,
) -> plt.Figure:
    subset = df[df["produto_padronizado"] == produto]
    ranking = (
        subset.groupby("estado_sigla")["preco_medio"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top = ranking.head(top_n)
    colors_top = sns.color_palette("Reds_r", n_colors=top_n)
    axes[0].barh(top.index[::-1], top.values[::-1], color=colors_top[::-1])
    axes[0].set_title(f"Top {top_n} Estados Mais Caros", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Preço Médio (R$/litro)")
    for i, v in enumerate(top.values[::-1]):
        axes[0].text(v + 0.02, i, f"R$ {v:.2f}", va="center", fontsize=9)

    bottom = ranking.tail(top_n)
    colors_bottom = sns.color_palette("Greens", n_colors=top_n)
    axes[1].barh(bottom.index[::-1], bottom.values[::-1], color=colors_bottom[::-1])
    axes[1].set_title(
        f"Top {top_n} Estados Mais Baratos", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Preço Médio (R$/litro)")
    for i, v in enumerate(bottom.values[::-1]):
        axes[1].text(v + 0.02, i, f"R$ {v:.2f}", va="center", fontsize=9)

    fig.suptitle(
        f"Comparativo de Preço por Estado — {produto}",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_bandeiras(
    df: pd.DataFrame,
    produto: str = "Gasolina Comum",
) -> plt.Figure:
    subset = df[df["produto_padronizado"] == produto]

    contagem = subset["bandeira"].value_counts()
    bandeiras_relevantes = contagem[contagem > 100].index
    subset = subset[subset["bandeira"].isin(bandeiras_relevantes)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ordem = (
        subset.groupby("bandeira")["preco_medio"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=subset,
        x="bandeira",
        y="preco_medio",
        order=ordem,
        hue="bandeira",
        hue_order=ordem,
        palette="Set2",
        dodge=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title(
        f"Distribuição de Preço por Bandeira — {produto}",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Bandeira", fontsize=12)
    ax.set_ylabel("Preço Médio (R$/litro)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_sazonalidade(
    df: pd.DataFrame,
    produto: str = "Gasolina Comum",
) -> plt.Figure:
    subset = df[df["produto_padronizado"] == produto]
    sazonal = subset.groupby("mes")["preco_medio"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sazonal.index, sazonal.values, color=sns.color_palette("coolwarm", 12))
    ax.set_title(
        f"Sazonalidade do Preço — {produto} (Média por Mês)",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Mês", fontsize=12)
    ax.set_ylabel("Preço Médio (R$/litro)", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("R$ %.2f"))
    plt.tight_layout()
    return fig


def plot_pp_vs_sp_vs_brasil(
    df: pd.DataFrame,
    produto: str = "Gasolina Comum",
) -> go.Figure:
    df = df[df["produto_padronizado"] == produto].copy()
    df["data"] = construir_coluna_data(df)
    df = df.sort_values("data")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["data"], y=df["preco_pp"],
        name="Presidente Prudente", mode="lines",
        line=dict(width=2.5, color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=df["data"], y=df["preco_sp"],
        name="São Paulo (Estado)", mode="lines",
        line=dict(width=2, color="#ff7f0e", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=df["data"], y=df["preco_brasil"],
        name="Brasil (Nacional)", mode="lines",
        line=dict(width=2, color="#2ca02c", dash="dot"),
    ))

    fig.update_layout(
        title=dict(
            text=f"Preço da {produto}: Presidente Prudente vs SP vs Brasil",
            font=dict(size=18),
        ),
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_previsto_vs_real(
    modelo_resultado: dict,
    df_features: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    n_splits: int = 5,
) -> go.Figure:
    feature_cols = modelo_resultado["features"]
    use_retorno = modelo_resultado.get("target_retorno", False)

    X = df_features[feature_cols]
    y_preco = df_features[coluna_preco]

    if use_retorno:
        y = df_features["variacao_pct"]
    else:
        y = y_preco

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    modelo_temp = clone(modelo_resultado["modelo"])
    modelo_temp.fit(X_train, y_train)
    y_pred_raw = modelo_temp.predict(X_test)

    if use_retorno:
        preco_hist = df_features[coluna_preco].values
        preco_inicial = preco_hist[test_idx[0] - 1] if test_idx[0] > 0 else preco_hist[0]
        precos_pred = []
        preco_atual = preco_inicial
        for variacao in y_pred_raw:
            preco_atual = preco_atual * (1 + variacao)
            precos_pred.append(preco_atual)
        y_pred_plot = np.array(precos_pred)
        y_real_plot = preco_hist[test_idx]
    else:
        y_pred_plot = y_pred_raw
        y_real_plot = y_preco.iloc[test_idx].values

    datas = construir_coluna_data(df_features.iloc[test_idx])
    mae = np.mean(np.abs(y_real_plot - y_pred_plot))
    nome_modelo = (
        type(modelo_resultado["modelo"]).__name__
        .replace("Regressor", "")
        .replace("Forest", " Forest")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datas, y=y_real_plot,
        name="Real", mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>Real</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=datas, y=y_pred_plot,
        name="Previsto", mode="lines+markers",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=5, symbol="x"),
        hovertemplate="<b>Previsto</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Previsto vs Real — {nome_modelo}  |  MAE: R$ {mae:.2f}  (out-of-sample)",
            font=dict(size=16),
        ),
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def plot_holt_winters_vs_real(
    modelo_resultado: dict,
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    n_splits: int = 5,
    max_train_size: int = 60,
    test_size: int = 12,
) -> go.Figure:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from src.ml import _serie_mensal_completa
    except ImportError:
        logger.warning("statsmodels não instalado — gráfico HW indisponível")
        return go.Figure()

    seasonal_periods = modelo_resultado.get("seasonal_periods", 12)
    serie = _serie_mensal_completa(df, coluna_preco)
    n_serie = len(serie)

    hw_test_size = test_size if n_serie >= (2 * seasonal_periods + n_splits * test_size) else None
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=hw_test_size)
    splits = list(tscv.split(serie))

    train_idx, test_idx = None, None
    for ti, vi in reversed(splits):
        if len(ti) >= 2 * seasonal_periods:
            train_idx, test_idx = ti, vi
            break

    if train_idx is None:
        logger.warning("Nenhum fold válido para plot_holt_winters_vs_real")
        return go.Figure()

    serie_train = serie.iloc[train_idx]
    y_test = serie.iloc[test_idx].values
    datas = serie.iloc[test_idx].index

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"statsmodels(\.|$)",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"statsmodels(\.|$)",
        )
        hw = ExponentialSmoothing(
            serie_train,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit(optimized=True, remove_bias=True)

    y_pred = hw.forecast(len(test_idx)).values
    mae = np.mean(np.abs(y_test - y_pred))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datas, y=y_test,
        name="Real", mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>Real</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=datas, y=y_pred,
        name="Previsto (HW)", mode="lines+markers",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=5, symbol="x"),
        hovertemplate="<b>Previsto (HW)</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Previsto vs Real — Holt-Winters  |  MAE: R$ {mae:.2f}  (out-of-sample)",
            font=dict(size=16),
        ),
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def plot_contrafactual(df_contrafactual: pd.DataFrame, ano_corte: int = 2019) -> go.Figure:
    df = df_contrafactual.copy()
    datas = df["data"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=datas,
        y=df["preco_hipotetico"],
        name=f"Hipotético (tendência pré-{ano_corte})",
        mode="lines",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        fill=None,
        hovertemplate="<b>Hipotético</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=datas,
        y=df["preco_real"],
        name="Preço real",
        mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.15)",
        hovertemplate="<b>Real</b><br>%{x|%b %Y}: R$ %{y:.2f}<extra></extra>",
    ))

    data_inicio_proj = pd.Timestamp(f"{ano_corte + 1}-01-01")
    fig.add_vline(
        x=data_inicio_proj.timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Início da projeção ({ano_corte + 1})",
        annotation_position="top right",
        annotation_font_size=11,
    )

    max_idx = df["diferenca"].abs().idxmax()
    mes_pico = df.loc[max_idx, "data"]
    preco_pico = df.loc[max_idx, "preco_real"]
    delta_pico = df.loc[max_idx, "diferenca"]
    sinal = "+" if delta_pico >= 0 else ""
    fig.add_annotation(
        x=mes_pico,
        y=preco_pico,
        text=f"Pico: {sinal}R${delta_pico:.2f}<br>{mes_pico.strftime('%b/%Y')}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#d62728",
        font=dict(size=11, color="#d62728"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d62728",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Preço Real vs Hipotético (modelo treinado até {ano_corte}) "
                "— Impacto de Choques Externos"
            ),
            font=dict(size=15),
        ),
        xaxis_title="Data",
        yaxis_title="Preço Médio (R$/litro)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def plot_correlacao_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = ["valor_venda", "dolar_venda", "preco_brent_usd"]
    cols_presentes = [c for c in cols if c in df.columns]

    if len(cols_presentes) < 2:
        logger.warning("Dados insuficientes para heatmap de correlação")
        return plt.figure()

    corr = df[cols_presentes].corr()

    labels = {
        "valor_venda": "Preço Combustível",
        "dolar_venda": "Dólar (USD/BRL)",
        "preco_brent_usd": "Petróleo Brent (USD)",
    }
    corr = corr.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, cmap="RdYlBu_r", vmin=-1, vmax=1,
        fmt=".3f", linewidths=0.5, ax=ax, square=True,
    )
    ax.set_title(
        "Correlação: Combustível vs Dólar vs Brent",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return fig
