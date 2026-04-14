import pandas as pd
import numpy as np
import pytest

from src.ml import (
    preparar_features_temporais,
    treinar_modelo_previsao,
    prever_proximo_periodo,
    clustering_municipios,
    treinar_holt_winters,
    prever_holt_winters,
    analise_contrafactual_holt_winters,
)


def criar_serie_temporal(n_meses=60):
    """Cria série temporal sintética para testes."""
    np.random.seed(42)
    datas = pd.date_range("2019-01-01", periods=n_meses, freq="MS")
    precos = 5.0 + np.cumsum(np.random.randn(n_meses) * 0.1)
    precos = np.maximum(precos, 2.0)

    return pd.DataFrame({
        "ano": datas.year,
        "mes": datas.month,
        "preco_medio": precos,
        "produto_padronizado": "Gasolina Comum",
    })


def criar_serie_sazonal(n_meses=60):
    datas = pd.date_range("2009-01-01", periods=n_meses, freq="MS")
    tendencia = np.linspace(5.0, 7.0, n_meses)
    sazonalidade = 0.3 * np.sin(2 * np.pi * np.arange(n_meses) / 12)
    ruido = np.random.randn(n_meses) * 0.04
    precos = np.maximum(tendencia + sazonalidade + ruido, 2.0)
    return pd.DataFrame({
        "ano": datas.year,
        "mes": datas.month,
        "preco_medio": precos,
        "produto_padronizado": "Gasolina Comum",
    })


def criar_dados_clustering(n_municipios=100):
    np.random.seed(42)
    return pd.DataFrame({
        "estado_sigla": np.repeat(
            np.random.choice(["SP", "RJ", "MG", "RS"], n_municipios), 50
        ),
        "municipio": np.repeat(
            [f"MUNICIPIO_{i}" for i in range(n_municipios)], 50
        ),
        "valor_venda": np.random.uniform(4.0, 8.0, n_municipios * 50),
    })


def criar_dados_clustering_pequeno():
    rows = []
    for uf, municipio, base in [("SP", "A", 5.0), ("SP", "B", 5.5), ("RJ", "C", 6.0)]:
        for i in range(60):
            rows.append({
                "estado_sigla": uf,
                "municipio": municipio,
                "valor_venda": base + (i % 5) * 0.01,
            })
    return pd.DataFrame(rows)


class TestFeaturesTemporais:
    def test_cria_lags(self):
        df = criar_serie_temporal()
        resultado = preparar_features_temporais(df, lags=6)
        for i in range(1, 7):
            assert f"lag_{i}" in resultado.columns

    def test_cria_medias_moveis(self):
        df = criar_serie_temporal()
        resultado = preparar_features_temporais(df)
        assert "media_movel_3m" in resultado.columns
        assert "media_movel_6m" in resultado.columns

    def test_cria_features_sazonais(self):
        df = criar_serie_temporal()
        resultado = preparar_features_temporais(df)
        assert "mes_sin" in resultado.columns
        assert "mes_cos" in resultado.columns

    def test_remove_nans(self):
        df = criar_serie_temporal()
        resultado = preparar_features_temporais(df)
        assert resultado.isna().sum().sum() == 0

    def test_menos_linhas_que_original(self):
        df = criar_serie_temporal(60)
        resultado = preparar_features_temporais(df, lags=6)
        assert len(resultado) < len(df)


class TestModeloPrevisao:
    def test_treina_random_forest(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df)
        resultado = treinar_modelo_previsao(
            df_features, modelo_tipo="random_forest", n_splits=3
        )
        assert "modelo" in resultado
        assert "metricas_cv" in resultado
        assert "importancia_features" in resultado

    def test_treina_gradient_boosting(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df)
        resultado = treinar_modelo_previsao(
            df_features, modelo_tipo="gradient_boosting", n_splits=3
        )
        assert "modelo" in resultado

    def test_metricas_razoaveis(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df)
        resultado = treinar_modelo_previsao(df_features, n_splits=3)
        assert resultado["metricas_cv"]["r2"]["media"] > -10
        assert resultado["metricas_cv"]["mae"]["media"] > 0


class TestPrevisao:
    def test_gera_previsao(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df)
        modelo_resultado = treinar_modelo_previsao(df_features, n_splits=3)
        previsoes = prever_proximo_periodo(modelo_resultado, df_features, meses_futuro=6)
        assert len(previsoes) == 6
        assert "preco_medio" in previsoes.columns
        assert "data" in previsoes.columns
        assert "variacao_prevista" in previsoes.columns

    def test_previsao_valores_positivos(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df)
        modelo_resultado = treinar_modelo_previsao(df_features, n_splits=3)
        previsoes = prever_proximo_periodo(modelo_resultado, df_features, meses_futuro=3)
        assert (previsoes["preco_medio"] > 0).all()

    def test_previsao_retorno_reconstroi_preco(self):
        df = criar_serie_temporal(60)
        df_features = preparar_features_temporais(df, target_retorno=True)
        modelo_resultado = treinar_modelo_previsao(df_features, n_splits=3, target_retorno=True)
        previsoes = prever_proximo_periodo(modelo_resultado, df_features, meses_futuro=3)
        ultimo_real = df_features["preco_medio"].iloc[-1]
        preco_1 = previsoes["preco_medio"].iloc[0]
        var_1 = previsoes["variacao_prevista"].iloc[0]
        assert abs(preco_1 - ultimo_real * (1 + var_1)) < 1e-6


class TestClustering:
    def test_clustering_basico(self):
        df = criar_dados_clustering()
        resultado = clustering_municipios(df, n_clusters=3)
        assert "modelo" in resultado
        assert "dados_clusterizados" in resultado
        assert "perfil_clusters" in resultado

    def test_n_clusters_correto(self):
        df = criar_dados_clustering()
        resultado = clustering_municipios(df, n_clusters=4)
        assert resultado["dados_clusterizados"]["cluster"].nunique() == 4

    def test_inercias_decrescentes(self):
        df = criar_dados_clustering()
        resultado = clustering_municipios(df)
        inercias = resultado["inercias"]
        assert inercias["inercia"].is_monotonic_decreasing

    def test_clustering_com_poucos_municipios(self):
        df = criar_dados_clustering_pequeno()
        resultado = clustering_municipios(df, n_clusters=2)
        assert len(resultado["dados_clusterizados"]) == 3
        assert resultado["dados_clusterizados"]["cluster"].nunique() == 2

    def test_n_clusters_maior_que_amostras(self):
        df = criar_dados_clustering_pequeno()
        resultado = clustering_municipios(df, n_clusters=5)
        assert resultado["modelo"].n_clusters == 3
        assert resultado["dados_clusterizados"]["cluster"].nunique() == 3

    def test_cotovelo_respeita_n_samples(self):
        df = criar_dados_clustering_pequeno()
        resultado = clustering_municipios(df, n_clusters=2)
        ks = resultado["inercias"]["k"].tolist()
        assert ks == [2, 3]


class TestHoltWinters:
    def test_treina_holt_winters(self):
        df = criar_serie_temporal(60)
        resultado = treinar_holt_winters(df, coluna_preco="preco_medio", n_splits=3)
        assert "modelo" in resultado
        assert "metricas_cv" in resultado
        assert resultado.get("tipo") == "holt_winters"

    def test_metricas_razoaveis(self):
        df = criar_serie_sazonal(60)
        resultado = treinar_holt_winters(df, coluna_preco="preco_medio", n_splits=3)
        assert resultado["metricas_cv"]["mae"]["media"] > 0
        assert resultado["metricas_cv"]["r2"]["media"] > 0

    def test_previsao_holt_winters(self):
        df = criar_serie_temporal(60)
        resultado = treinar_holt_winters(df, coluna_preco="preco_medio", n_splits=3)
        previsoes = prever_holt_winters(resultado, meses_futuro=6)
        assert len(previsoes) == 6
        assert "preco_medio" in previsoes.columns
        assert "data" in previsoes.columns
        assert "variacao_prevista" in previsoes.columns

    def test_previsao_valores_positivos(self):
        df = criar_serie_temporal(60)
        resultado = treinar_holt_winters(df, coluna_preco="preco_medio", n_splits=3)
        previsoes = prever_holt_winters(resultado, meses_futuro=3)
        assert (previsoes["preco_medio"] > 0).all()

    def test_serie_muito_curta_levanta_erro(self):
        df = criar_serie_temporal(n_meses=20)
        with pytest.raises(ValueError, match="muito curta"):
            treinar_holt_winters(df, coluna_preco="preco_medio", n_splits=3)


class TestFeaturesTemporaisEdgeCases:
    def test_serie_muito_curta_para_lags(self):
        df = criar_serie_temporal(n_meses=5)
        resultado = preparar_features_temporais(df, lags=6)
        assert resultado.empty

    def test_preco_constante(self):
        df = criar_serie_temporal(60)
        df["preco_medio"] = 5.0
        resultado = preparar_features_temporais(df)
        assert resultado["variacao_pct"].dropna().eq(0).all()

    def test_lags_customizados(self):
        df = criar_serie_temporal(60)
        resultado = preparar_features_temporais(df, lags=3)
        assert "lag_3" in resultado.columns
        assert "lag_4" not in resultado.columns


class TestContrafactual:
    def test_analise_contrafactual_executa(self):
        df = criar_serie_sazonal(120)
        resultado = analise_contrafactual_holt_winters(df, ano_corte=2015)
        assert "df_contrafactual" in resultado
        assert "resumo" in resultado
        assert "preco_real" in resultado["df_contrafactual"].columns
        assert "preco_hipotetico" in resultado["df_contrafactual"].columns

    def test_contrafactual_tem_diferencas(self):
        df = criar_serie_sazonal(120)
        resultado = analise_contrafactual_holt_winters(df, ano_corte=2015)
        df_cf = resultado["df_contrafactual"]
        assert len(df_cf) > 0
        assert "max_diferenca_absoluta" in resultado["resumo"]
        assert "max_diferenca_pct" in resultado["resumo"]
        assert "media_diferenca" in resultado["resumo"]
        assert "mes_maior_impacto" in resultado["resumo"]

    def test_contrafactual_periodo_correto(self):
        df = criar_serie_sazonal(120)
        resultado = analise_contrafactual_holt_winters(df, ano_corte=2015)
        df_cf = resultado["df_contrafactual"]
        assert df_cf["data"].min().year == 2016
        assert df_cf["data"].min().month == 1

    def test_contrafactual_serie_muito_curta_levanta_erro(self):
        df = criar_serie_sazonal(36)
        with pytest.raises(ValueError, match="Sem dados após"):
            analise_contrafactual_holt_winters(df, ano_corte=2015)
