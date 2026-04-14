from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from src.utils import DATA_EXTERNAL, setup_logger

logger = setup_logger("scraping")


def obter_cotacao_dolar(
    data_inicio: str = "01-01-2004",
    data_fim: Optional[str] = None,
) -> pd.DataFrame:
    if data_fim is None:
        data_fim = datetime.now().strftime("%m-%d-%Y")

    logger.info(f"Buscando cotação do dólar de {data_inicio} a {data_fim}...")

    url = (
        "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/"
        "odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,"
        "dataFinalCotacao=@dataFinalCotacao)"
    )

    dt_inicio = datetime.strptime(data_inicio, "%m-%d-%Y")
    dt_fim = datetime.strptime(data_fim, "%m-%d-%Y")
    all_data = []

    ano_atual = dt_inicio.year
    while ano_atual <= dt_fim.year:
        inicio_ano = f"01-01-{ano_atual}"
        fim_ano = f"12-31-{ano_atual}"
        if ano_atual == dt_inicio.year:
            inicio_ano = data_inicio
        if ano_atual == dt_fim.year:
            fim_ano = data_fim

        params = {
            "@dataInicial": f"'{inicio_ano}'",
            "@dataFinalCotacao": f"'{fim_ano}'",
            "$format": "json",
            "$top": "10000",
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get("value", [])
            all_data.extend(data)
            logger.info(f"  {ano_atual}: {len(data)} registros")
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning(f"  {ano_atual}: erro — {e}")

        ano_atual += 1

    if not all_data:
        logger.warning("Nenhum dado de cotação retornado pela API")
        return pd.DataFrame(columns=["data", "cotacao_compra", "cotacao_venda"])

    df = pd.DataFrame(all_data)
    df["data"] = pd.to_datetime(df["dataHoraCotacao"]).dt.date
    df = df.rename(columns={
        "cotacaoCompra": "cotacao_compra",
        "cotacaoVenda": "cotacao_venda",
    })
    df = df[["data", "cotacao_compra", "cotacao_venda"]]
    df = df.drop_duplicates(subset=["data"], keep="last")
    df = df.sort_values("data").reset_index(drop=True)
    df["data"] = pd.to_datetime(df["data"])

    logger.info(f"Cotação do dólar: {len(df)} dias obtidos")
    return df


def obter_preco_brent() -> pd.DataFrame:
    logger.info("Buscando preço histórico do Brent (EIA)...")

    url = (
        "https://www.eia.gov/dnav/pet/hist_xls/"
        "RBRTEd.xls"
    )

    try:
        df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
        df.columns = ["data", "preco_brent_usd"]
        df["data"] = pd.to_datetime(df["data"])
        df = df.dropna(subset=["preco_brent_usd"])
        df = df.sort_values("data").reset_index(drop=True)
        logger.info(f"Preço Brent: {len(df)} registros obtidos")
        return df
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.warning(f"Falha ao obter dados do EIA: {e}")
        logger.info("Tentando fonte alternativa (FRED)...")
        return _obter_brent_fred()


def _obter_brent_fred() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

    try:
        df = pd.read_csv(url)
        df.columns = ["data", "preco_brent_usd"]
        df["data"] = pd.to_datetime(df["data"])
        df["preco_brent_usd"] = pd.to_numeric(
            df["preco_brent_usd"], errors="coerce"
        )
        df = df.dropna(subset=["preco_brent_usd"])
        df = df.sort_values("data").reset_index(drop=True)
        logger.info(f"Preço Brent (FRED): {len(df)} registros obtidos")
        return df
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Falha ao obter dados do FRED: {e}")
        return pd.DataFrame(columns=["data", "preco_brent_usd"])


def salvar_dados_externos(
    df_dolar: Optional[pd.DataFrame] = None,
    df_brent: Optional[pd.DataFrame] = None,
) -> dict:
    paths = {}

    if df_dolar is None:
        df_dolar = obter_cotacao_dolar()
    path_dolar = DATA_EXTERNAL / "cotacao_dolar.parquet"
    df_dolar.to_parquet(path_dolar, index=False)
    paths["dolar"] = str(path_dolar)
    logger.info(f"Dólar salvo: {path_dolar}")

    if df_brent is None:
        df_brent = obter_preco_brent()
    path_brent = DATA_EXTERNAL / "preco_brent.parquet"
    df_brent.to_parquet(path_brent, index=False)
    paths["brent"] = str(path_brent)
    logger.info(f"Brent salvo: {path_brent}")

    return paths
