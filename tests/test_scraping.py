from unittest.mock import patch, MagicMock

import pandas as pd
import requests

from src.scraping import obter_cotacao_dolar, obter_preco_brent, _obter_brent_fred


def _mock_bcb_response(data):
    mock = MagicMock()
    mock.json.return_value = {"value": data}
    mock.raise_for_status.return_value = None
    return mock


def _cotacao_fake(data_str="2023-01-02T13:00:00.000-0300"):
    return {
        "dataHoraCotacao": data_str,
        "cotacaoCompra": 5.10,
        "cotacaoVenda": 5.12,
    }


class TestObterCotacaoDolar:
    @patch("src.scraping.requests.get")
    def test_retorna_dataframe(self, mock_get):
        mock_get.return_value = _mock_bcb_response([_cotacao_fake()])
        df = obter_cotacao_dolar("01-01-2023", "12-31-2023")
        assert isinstance(df, pd.DataFrame)
        assert "cotacao_venda" in df.columns

    @patch("src.scraping.requests.get")
    def test_retorna_vazio_quando_api_falha(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
        df = obter_cotacao_dolar("01-01-2023", "12-31-2023")
        assert df.empty

    @patch("src.scraping.requests.get")
    def test_deduplica_por_data(self, mock_get):
        duplicado = [_cotacao_fake(), _cotacao_fake()]
        mock_get.return_value = _mock_bcb_response(duplicado)
        df = obter_cotacao_dolar("01-01-2023", "12-31-2023")
        assert df["data"].duplicated().sum() == 0

    @patch("src.scraping.requests.get")
    def test_colunas_esperadas(self, mock_get):
        mock_get.return_value = _mock_bcb_response([_cotacao_fake()])
        df = obter_cotacao_dolar("01-01-2023", "12-31-2023")
        assert set(df.columns) == {"data", "cotacao_compra", "cotacao_venda"}


class TestObterPrecoBrent:
    @patch("src.scraping.pd.read_excel")
    def test_usa_eia_quando_disponivel(self, mock_excel):
        mock_excel.return_value = pd.DataFrame({
            "data": pd.date_range("2023-01-01", periods=5),
            "preco_brent_usd": [80.0, 81.0, 79.5, 82.0, 83.5],
        })
        df = obter_preco_brent()
        assert "preco_brent_usd" in df.columns
        assert len(df) == 5

    @patch("src.scraping._obter_brent_fred")
    @patch("src.scraping.pd.read_excel")
    def test_fallback_para_fred(self, mock_excel, mock_fred):
        mock_excel.side_effect = requests.exceptions.RequestException("timeout")
        mock_fred.return_value = pd.DataFrame({
            "data": pd.date_range("2023-01-01", periods=3),
            "preco_brent_usd": [80.0, 81.0, 82.0],
        })
        df = obter_preco_brent()
        mock_fred.assert_called_once()
        assert "preco_brent_usd" in df.columns


class TestObterBrentFred:
    @patch("src.scraping.pd.read_csv")
    def test_retorna_vazio_quando_falha(self, mock_csv):
        mock_csv.side_effect = requests.exceptions.ConnectionError("timeout")
        df = _obter_brent_fred()
        assert df.empty
        assert "preco_brent_usd" in df.columns

    @patch("src.scraping.pd.read_csv")
    def test_remove_valores_nulos(self, mock_csv):
        mock_csv.return_value = pd.DataFrame({
            "data": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "preco_brent_usd": [80.0, None, 82.0],
        })
        df = _obter_brent_fred()
        assert df["preco_brent_usd"].notna().all()
