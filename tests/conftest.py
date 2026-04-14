import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def serie_temporal():
    np.random.seed(42)
    datas = pd.date_range("2019-01-01", periods=60, freq="MS")
    precos = 5.0 + np.cumsum(np.random.randn(60) * 0.1)
    return pd.DataFrame({
        "ano": datas.year,
        "mes": datas.month,
        "preco_medio": np.maximum(precos, 2.0),
        "produto_padronizado": "Gasolina Comum",
    })


@pytest.fixture
def df_municipios():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "estado_sigla": np.repeat(np.random.choice(["SP", "RJ", "MG", "RS"], n), 50),
        "municipio": np.repeat([f"MUNICIPIO_{i}" for i in range(n)], 50),
        "valor_venda": np.random.uniform(4.0, 8.0, n * 50),
    })


@pytest.fixture
def df_etl_base():
    return pd.DataFrame({
        "data_coleta": ["01/01/2023", "15/01/2023", "30/01/2023", None],
        "valor_venda": ["5,50", "5,60", "999", "5,70"],
        "valor_compra": ["4,50", "4,60", None, "4,70"],
        "produto": ["GASOLINA COMUM", "GASOLINA COMUM", "GASOLINA COMUM", "ETANOL"],
        "estado": ["SAO PAULO", "SAO PAULO", "RIO DE JANEIRO", "SAO PAULO"],
        "municipio": ["SAO PAULO", "CAMPINAS", "RIO DE JANEIRO", "SAO PAULO"],
        "bandeira": ["SHELL", "IPIRANGA", "BRANCA", "BR"],
    })
