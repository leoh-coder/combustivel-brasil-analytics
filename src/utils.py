import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "sa-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "combustivel-brasil-analytics")


def setup_dirs():
    for p in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def construir_coluna_data(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["ano"].astype(int).astype(str) + "-"
        + df["mes"].astype(int).astype(str).str.zfill(2) + "-01"
    )


PRODUTOS_MAP = {
    "GASOLINA": "Gasolina Comum",
    "GASOLINA COMUM": "Gasolina Comum",
    "GASOLINA ADITIVADA": "Gasolina Aditivada",
    "ETANOL HIDRATADO": "Etanol",
    "ETANOL": "Etanol",
    "OLEO DIESEL": "Diesel",
    "DIESEL": "Diesel",
    "OLEO DIESEL S10": "Diesel S10",
    "DIESEL S10": "Diesel S10",
    "DIESEL S50": "Diesel S50",
    "GNV": "GNV",
    "GLP": "GLP",
}

ESTADO_PARA_SIGLA = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "GOIAS": "GO", "MARANHAO": "MA",
    "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG", "PARA": "PA", "PARAIBA": "PB",
    "PARANA": "PR", "PERNAMBUCO": "PE", "PIAUI": "PI",
    "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO", "RORAIMA": "RR",
    "SANTA CATARINA": "SC", "SAO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}
setup_dirs()
