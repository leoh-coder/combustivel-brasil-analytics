import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import setup_logger, construir_coluna_data

logger = setup_logger("ml")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def preparar_features_temporais(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    lags: int = 6,
    target_retorno: bool = True,
) -> pd.DataFrame:
    df = df.sort_values(["ano", "mes"]).copy()

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df[coluna_preco].shift(i)

    df["media_movel_3m"] = df[coluna_preco].rolling(3).mean()
    df["media_movel_6m"] = df[coluna_preco].rolling(6).mean()

    df["variacao_pct"] = df[coluna_preco].pct_change()
    df["variacao_pct_3m"] = df[coluna_preco].pct_change(3)

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    df = df.dropna()
    return df


def treinar_modelo_previsao(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    modelo_tipo: str = "random_forest",
    n_splits: int = 5,
    target_retorno: bool = True,
) -> dict:

    logger.info(f"Treinando modelo de previsão ({modelo_tipo})...")

    feature_cols = [
        c for c in df.columns
        if c.startswith(("lag_", "media_movel_", "variacao_pct", "mes_"))
    ]
    if "dolar_venda" in df.columns:
        feature_cols.append("dolar_venda")
    if "preco_brent_usd" in df.columns:
        feature_cols.append("preco_brent_usd")

    if target_retorno and "variacao_pct" in df.columns:
        feature_cols = [c for c in feature_cols if c != "variacao_pct"]
        y = df["variacao_pct"]
    else:
        y = df[coluna_preco]

    X = df[feature_cols]

    if modelo_tipo == "gradient_boosting":
        modelo = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
    else:
        modelo = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metricas_cv = {"mae": [], "rmse": [], "r2": []}

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        metricas_cv["mae"].append(mean_absolute_error(y_test, y_pred))
        metricas_cv["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metricas_cv["r2"].append(r2_score(y_test, y_pred))

    modelo.fit(X, y)

    importancia = pd.DataFrame({
        "feature": feature_cols,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False)

    resultados = {
        "modelo": modelo,
        "features": feature_cols,
        "target_retorno": target_retorno,
        "coluna_preco": coluna_preco,
        "metricas_cv": {
            k: {"media": np.mean(v), "std": np.std(v)}
            for k, v in metricas_cv.items()
        },
        "importancia_features": importancia,
    }

    mae = resultados["metricas_cv"]["mae"]["media"]
    r2 = resultados["metricas_cv"]["r2"]["media"]
    escala = "variação %" if target_retorno else "R$"
    logger.info(f"Modelo treinado — MAE médio: {mae:.4f} ({escala}), R² médio: {r2:.4f}")

    return resultados


def treinar_baselines(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    n_splits: int = 5,
    target_retorno: bool = True,
) -> list:
    feature_cols = [
        c for c in df.columns
        if c.startswith(("lag_", "media_movel_", "variacao_pct", "mes_"))
    ]
    if "dolar_venda" in df.columns:
        feature_cols.append("dolar_venda")
    if "preco_brent_usd" in df.columns:
        feature_cols.append("preco_brent_usd")

    if target_retorno and "variacao_pct" in df.columns:
        feature_cols = [c for c in feature_cols if c != "variacao_pct"]
        y = df["variacao_pct"]
        naive_series = df["variacao_pct"].shift(1).fillna(0)
        mm3_series = df["variacao_pct"].rolling(3).mean().shift(1).fillna(0)
    else:
        y = df[coluna_preco]
        naive_series = df[coluna_preco].shift(1).fillna(method="bfill")
        mm3_series = df[coluna_preco].rolling(3).mean().shift(1).fillna(method="bfill")

    X = df[feature_cols]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    resultados = []

    metricas = {"mae": [], "rmse": [], "r2": []}
    for _, test_idx in tscv.split(X):
        y_test = y.iloc[test_idx]
        y_pred = naive_series.iloc[test_idx]
        metricas["mae"].append(mean_absolute_error(y_test, y_pred))
        metricas["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metricas["r2"].append(r2_score(y_test, y_pred))
    nome_naive = "Naive (última variação)" if target_retorno else "Naive (lag_1)"
    resultados.append({
        "nome": nome_naive,
        "metricas_cv": {k: {"media": np.mean(v), "std": np.std(v)} for k, v in metricas.items()},
    })

    metricas = {"mae": [], "rmse": [], "r2": []}
    for _, test_idx in tscv.split(X):
        y_test = y.iloc[test_idx]
        y_pred = mm3_series.iloc[test_idx]
        metricas["mae"].append(mean_absolute_error(y_test, y_pred))
        metricas["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metricas["r2"].append(r2_score(y_test, y_pred))
    nome_mm = "MM3m (variações)" if target_retorno else "Média Móvel 3m"
    resultados.append({
        "nome": nome_mm,
        "metricas_cv": {k: {"media": np.mean(v), "std": np.std(v)} for k, v in metricas.items()},
    })

    metricas = {"mae": [], "rmse": [], "r2": []}
    lr = LinearRegression()
    for train_idx, test_idx in tscv.split(X):
        lr.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = lr.predict(X.iloc[test_idx])
        metricas["mae"].append(mean_absolute_error(y.iloc[test_idx], y_pred))
        metricas["rmse"].append(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred)))
        metricas["r2"].append(r2_score(y.iloc[test_idx], y_pred))
    resultados.append({
        "nome": "Regressão Linear",
        "metricas_cv": {k: {"media": np.mean(v), "std": np.std(v)} for k, v in metricas.items()},
    })

    return resultados


def _serie_mensal_completa(df: pd.DataFrame, coluna_preco: str) -> pd.Series:
    df = df.sort_values(["ano", "mes"]).copy()
    datas = pd.to_datetime(
        df[["ano", "mes"]].assign(day=1).rename(columns={"ano": "year", "mes": "month"})
    )
    serie_bruta = pd.Series(df[coluna_preco].values, index=datas.values)
    idx_completo = pd.date_range(start=serie_bruta.index.min(), end=serie_bruta.index.max(), freq="MS")
    return serie_bruta.reindex(idx_completo).interpolate(method="linear")


def treinar_holt_winters(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    seasonal_periods: int = 12,
    n_splits: int = 5,
    max_train_meses: int = 60,
) -> dict:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        raise ImportError(
            "statsmodels não instalado. Execute: pip install statsmodels>=0.14.0"
        ) from e

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
    logger.info("Treinando Holt-Winters (trend=add, damped=True, seasonal=add)...")

    serie = _serie_mensal_completa(df, coluna_preco)

    n_serie = len(serie)
    hw_test_size = 12 if n_serie >= (2 * seasonal_periods + n_splits * 12) else None
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_meses, test_size=hw_test_size)
    metricas_cv: dict = {"mae": [], "rmse": [], "r2": []}

    for train_idx, test_idx in tscv.split(serie):
        if len(train_idx) < 2 * seasonal_periods:
            logger.debug(
                f"  Fold pulado: {len(train_idx)} meses < 2×{seasonal_periods} ciclos sazonais"
            )
            continue

        serie_train = serie.iloc[train_idx]
        y_test = serie.iloc[test_idx].values

        hw_cv = ExponentialSmoothing(
            serie_train,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit(optimized=True, remove_bias=True)

        y_pred = hw_cv.forecast(len(test_idx)).values
        metricas_cv["mae"].append(mean_absolute_error(y_test, y_pred))
        metricas_cv["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metricas_cv["r2"].append(r2_score(y_test, y_pred))

    if not metricas_cv["mae"]:
        raise ValueError(
            f"Todos os folds foram pulados — série muito curta para Holt-Winters "
            f"(mínimo: {2 * seasonal_periods} meses de treino)."
        )

    modelo_final = ExponentialSmoothing(
        serie,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True, remove_bias=True)

    resultado = {
        "modelo": modelo_final,
        "tipo": "holt_winters",
        "coluna_preco": coluna_preco,
        "seasonal_periods": seasonal_periods,
        "metricas_cv": {
            k: {"media": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in metricas_cv.items()
        },
    }

    mae = resultado["metricas_cv"]["mae"]["media"]
    r2 = resultado["metricas_cv"]["r2"]["media"]
    logger.info(f"Holt-Winters treinado — MAE médio: R$ {mae:.4f}, R² médio: {r2:.4f}")
    return resultado


def prever_holt_winters(
    modelo_resultado: dict,
    meses_futuro: int = 6,
) -> pd.DataFrame:
    modelo = modelo_resultado["modelo"]
    y_futuro = modelo.forecast(meses_futuro)

    preco_anterior = float(modelo.fittedvalues.iloc[-1])

    previsoes = []
    for data, preco in zip(y_futuro.index, y_futuro.values):
        preco = max(float(preco), 0.01)
        variacao = (preco - preco_anterior) / preco_anterior if preco_anterior > 0 else 0.0
        previsoes.append({
            "ano": data.year,
            "mes": data.month,
            "preco_medio": preco,
            "variacao_prevista": variacao,
            "tipo": "previsao",
        })
        preco_anterior = preco

    resultado = pd.DataFrame(previsoes)
    resultado["data"] = construir_coluna_data(resultado)

    logger.info(f"Previsão Holt-Winters gerada para {meses_futuro} meses")
    return resultado


def comparar_modelos(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    n_splits: int = 5,
    target_retorno: bool = True,
    df_original: pd.DataFrame = None,
) -> pd.DataFrame:
    logger.info("Comparando modelos (baselines + RF + GB + Holt-Winters)...")
    linhas = []

    escala_ml = "Variação %" if target_retorno else "Preço (R$)"

    for b in treinar_baselines(df, coluna_preco=coluna_preco, n_splits=n_splits, target_retorno=target_retorno):
        m = b["metricas_cv"]
        linhas.append({
            "Modelo": b["nome"],
            "Escala": escala_ml,
            "MAE (média)": round(m["mae"]["media"], 4),
            "MAE (std)": round(m["mae"]["std"], 4),
            "RMSE (média)": round(m["rmse"]["media"], 4),
            "RMSE (std)": round(m["rmse"]["std"], 4),
            "R² (média)": round(m["r2"]["media"], 4),
            "R² (std)": round(m["r2"]["std"], 4),
        })

    for tipo, nome in [("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]:
        r = treinar_modelo_previsao(
            df, coluna_preco=coluna_preco, modelo_tipo=tipo,
            n_splits=n_splits, target_retorno=target_retorno,
        )
        m = r["metricas_cv"]
        linhas.append({
            "Modelo": nome,
            "Escala": escala_ml,
            "MAE (média)": round(m["mae"]["media"], 4),
            "MAE (std)": round(m["mae"]["std"], 4),
            "RMSE (média)": round(m["rmse"]["media"], 4),
            "RMSE (std)": round(m["rmse"]["std"], 4),
            "R² (média)": round(m["r2"]["media"], 4),
            "R² (std)": round(m["r2"]["std"], 4),
        })

    df_hw = df_original if df_original is not None else df
    try:
        hw = treinar_holt_winters(df_hw, coluna_preco=coluna_preco, n_splits=n_splits)
        m = hw["metricas_cv"]
        linhas.append({
            "Modelo": "Holt-Winters",
            "Escala": "Preço (R$)",
            "MAE (média)": round(m["mae"]["media"], 4),
            "MAE (std)": round(m["mae"]["std"], 4),
            "RMSE (média)": round(m["rmse"]["media"], 4),
            "RMSE (std)": round(m["rmse"]["std"], 4),
            "R² (média)": round(m["r2"]["media"], 4),
            "R² (std)": round(m["r2"]["std"], 4),
        })
    except (ImportError, ValueError) as e:
        logger.warning(f"Holt-Winters falhou na comparação: {e}")

    df_cmp = pd.DataFrame(linhas)
    logger.info("Comparação concluída")
    return df_cmp


def prever_proximo_periodo(
    modelo_resultado: dict,
    df_historico: pd.DataFrame,
    meses_futuro: int = 6,
    coluna_preco: str = "preco_medio",
    target_retorno: bool = True,
) -> pd.DataFrame:
    modelo = modelo_resultado["modelo"]
    feature_cols = modelo_resultado["features"]
    use_retorno = modelo_resultado.get("target_retorno", target_retorno)
    coluna_preco = modelo_resultado.get("coluna_preco", coluna_preco)

    df = df_historico.copy()
    preco_atual = df[coluna_preco].iloc[-1]

    previsoes = []
    for _ in range(meses_futuro):
        ultimo = df.iloc[-1]
        proximo_mes = int(ultimo["mes"]) + 1
        proximo_ano = int(ultimo["ano"])
        if proximo_mes > 12:
            proximo_mes = 1
            proximo_ano += 1

        nova_linha = {"ano": proximo_ano, "mes": proximo_mes}

        n_lags = sum(1 for c in feature_cols if c.startswith("lag_"))
        for i in range(1, n_lags + 1):
            col = f"lag_{i}"
            if col in feature_cols:
                nova_linha[col] = df[coluna_preco].iloc[-i] if i <= len(df) else df[coluna_preco].iloc[0]

        precos_recentes = df[coluna_preco].values
        if "media_movel_3m" in feature_cols:
            nova_linha["media_movel_3m"] = precos_recentes[-3:].mean()
        if "media_movel_6m" in feature_cols:
            nova_linha["media_movel_6m"] = precos_recentes[-6:].mean()
        if "variacao_pct" in feature_cols:
            nova_linha["variacao_pct"] = (
                (precos_recentes[-1] - precos_recentes[-2]) / precos_recentes[-2]
            )
        if "variacao_pct_3m" in feature_cols:
            nova_linha["variacao_pct_3m"] = (
                (precos_recentes[-1] - precos_recentes[-4]) / precos_recentes[-4]
            )
        nova_linha["mes_sin"] = np.sin(2 * np.pi * proximo_mes / 12)
        nova_linha["mes_cos"] = np.cos(2 * np.pi * proximo_mes / 12)

        for col in ["dolar_venda", "preco_brent_usd"]:
            if col in feature_cols:
                nova_linha[col] = df[col].iloc[-1]

        X_pred = pd.DataFrame([nova_linha])[feature_cols]
        predicao = modelo.predict(X_pred)[0]

        if use_retorno:
            variacao_prevista = predicao
            preco_previsto = preco_atual * (1 + variacao_prevista)
            nova_linha["variacao_prevista"] = variacao_prevista
        else:
            preco_previsto = predicao
            nova_linha["variacao_prevista"] = (
                (preco_previsto - preco_atual) / preco_atual if preco_atual else 0
            )

        nova_linha[coluna_preco] = preco_previsto
        nova_linha["tipo"] = "previsao"
        preco_atual = preco_previsto

        previsoes.append(nova_linha)

        nova_df = pd.DataFrame([nova_linha])
        for c in df.columns:
            if c not in nova_df.columns:
                nova_df[c] = np.nan
        df = pd.concat([df, nova_df[df.columns]], ignore_index=True)

    resultado = pd.DataFrame(previsoes)
    resultado["data"] = construir_coluna_data(resultado)

    logger.info(f"Previsão gerada para {meses_futuro} meses")
    return resultado


def clustering_municipios(
    df: pd.DataFrame,
    n_clusters: int = 5,
) -> dict:
    logger.info(f"Clustering de municípios (k={n_clusters})...")
    if n_clusters < 1:
        raise ValueError("n_clusters deve ser >= 1")

    agg = df.groupby(["estado_sigla", "municipio"]).agg(
        preco_medio=("valor_venda", "mean"),
        preco_std=("valor_venda", "std"),
        preco_min=("valor_venda", "min"),
        preco_max=("valor_venda", "max"),
        n_registros=("valor_venda", "count"),
    ).reset_index()

    agg = agg[agg["n_registros"] >= 50]
    n_samples = len(agg)
    if n_samples == 0:
        raise ValueError("Sem municípios suficientes após filtro (mínimo de 50 registros).")

    n_clusters_ajustado = min(n_clusters, n_samples)
    if n_clusters_ajustado != n_clusters:
        logger.warning(
            f"n_clusters={n_clusters} maior que n_samples={n_samples}; usando k={n_clusters_ajustado}."
        )

    agg["amplitude"] = agg["preco_max"] - agg["preco_min"]
    agg["coef_variacao"] = agg["preco_std"] / agg["preco_medio"]

    feature_cols = ["preco_medio", "preco_std", "amplitude", "coef_variacao"]
    X = agg[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters_ajustado, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    perfil = agg.groupby("cluster")[feature_cols].mean().round(4)
    perfil["n_municipios"] = agg.groupby("cluster").size()

    inercias = []
    if n_samples >= 2:
        for k in range(2, min(10, n_samples) + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inercias.append({"k": k, "inercia": km.inertia_})

    resultado = {
        "modelo": kmeans,
        "scaler": scaler,
        "dados_clusterizados": agg,
        "perfil_clusters": perfil,
        "inercias": pd.DataFrame(inercias),
        "features": feature_cols,
    }

    logger.info(f"Clustering concluído — {n_clusters_ajustado} clusters formados")
    return resultado


def analise_contrafactual_holt_winters(
    df: pd.DataFrame,
    coluna_preco: str = "preco_medio",
    ano_corte: int = 2019,
    seasonal_periods: int = 12,
) -> dict:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        raise ImportError(
            "statsmodels não instalado. Execute: pip install statsmodels>=0.14.0"
        ) from e

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
    logger.info(f"Análise contrafactual Holt-Winters (corte: {ano_corte})...")

    serie = _serie_mensal_completa(df, coluna_preco)

    corte_data = pd.Timestamp(f"{ano_corte}-12-01")
    serie_treino = serie[serie.index <= corte_data]
    serie_futuro = serie[serie.index > corte_data]

    if len(serie_treino) < 2 * seasonal_periods:
        raise ValueError(
            f"Dados insuficientes até {ano_corte} para treinar Holt-Winters "
            f"(mínimo: {2 * seasonal_periods} meses). "
            f"Série disponível: {len(serie_treino)} meses."
        )
    if len(serie_futuro) == 0:
        raise ValueError(f"Sem dados após {ano_corte} na série — nada para comparar.")

    modelo = ExponentialSmoothing(
        serie_treino,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True, remove_bias=True)

    n_futuro = len(serie_futuro)
    y_hipotetico = modelo.forecast(n_futuro)

    df_cf = pd.DataFrame({
        "data": serie_futuro.index,
        "preco_real": serie_futuro.values,
        "preco_hipotetico": y_hipotetico.values,
    })
    df_cf["diferenca"] = df_cf["preco_real"] - df_cf["preco_hipotetico"]
    df_cf["diferenca_pct"] = df_cf["diferenca"] / df_cf["preco_hipotetico"].abs() * 100
    df_cf = df_cf.reset_index(drop=True)

    max_idx = df_cf["diferenca"].abs().idxmax()
    mes_maior = df_cf.loc[max_idx, "data"].strftime("%b/%Y")
    resumo = {
        "max_diferenca_absoluta": round(float(df_cf["diferenca"].abs().max()), 4),
        "max_diferenca_pct": round(float(df_cf["diferenca_pct"].abs().max()), 2),
        "media_diferenca": round(float(df_cf["diferenca"].mean()), 4),
        "mes_maior_impacto": mes_maior,
    }

    logger.info(
        f"Contrafactual concluído — maior impacto: {mes_maior} "
        f"(Δ R$ {resumo['max_diferenca_absoluta']:.2f})"
    )
    return {
        "df_contrafactual": df_cf,
        "ano_corte": ano_corte,
        "resumo": resumo,
    }


def analisar_correlacao(df: pd.DataFrame) -> dict:
    logger.info("Analisando correlação combustível vs câmbio vs petróleo...")

    cols = ["valor_venda", "dolar_venda", "preco_brent_usd"]
    cols_presentes = [c for c in cols if c in df.columns]

    if len(cols_presentes) < 2:
        logger.warning("Dados insuficientes para análise de correlação")
        return {}

    corr_geral = df[cols_presentes].corr()

    corr_por_ano = []
    if "ano" in df.columns:
        for ano in sorted(df["ano"].unique()):
            subset = df[df["ano"] == ano]
            if len(subset) > 30:
                corr = subset[cols_presentes].corr()
                for i, col1 in enumerate(cols_presentes):
                    for j, col2 in enumerate(cols_presentes):
                        if i < j:
                            corr_por_ano.append({
                                "ano": ano,
                                "var1": col1,
                                "var2": col2,
                                "correlacao": corr.loc[col1, col2],
                            })

    corr_lag = []
    if "dolar_venda" in df.columns:
        mensal = df.groupby(["ano", "mes"]).agg(
            preco_medio=("valor_venda", "mean"),
            dolar_medio=("dolar_venda", "mean"),
        ).reset_index().sort_values(["ano", "mes"])

        for lag in range(0, 7):
            mensal[f"dolar_lag_{lag}"] = mensal["dolar_medio"].shift(lag)
            corr_val = mensal["preco_medio"].corr(mensal[f"dolar_lag_{lag}"])
            corr_lag.append({"lag_meses": lag, "correlacao": corr_val})

    resultado = {
        "correlacao_geral": corr_geral,
        "correlacao_por_ano": pd.DataFrame(corr_por_ano),
        "correlacao_lag_dolar": pd.DataFrame(corr_lag),
    }

    return resultado
