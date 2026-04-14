import re
import unicodedata
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils import (
    DATA_RAW,
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    S3_BUCKET,
    setup_logger,
)

logger = setup_logger("ingestao")

ANP_BASE_URL = (
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/"
    "dados-abertos/serie-historica-de-precos-de-combustiveis"
)


def descobrir_links_csv(url: str = ANP_BASE_URL) -> list:
    logger.info("Buscando links de CSV na página da ANP...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.endswith(".csv") or "csv" in href.lower():
            nome = a_tag.get_text(strip=True) or href.split("/")[-1]
            semestre_match = re.search(r"(\d{4})", nome) or re.search(
                r"(\d{4})", href
            )
            semestre = semestre_match.group(1) if semestre_match else "desconhecido"
            links.append({"url": href, "nome": nome, "semestre": semestre})

    logger.info(f"Encontrados {len(links)} arquivos CSV")
    return links


def _normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def filtrar_links_csv(
    links: list,
    ano_min: int = 2004,
    ano_max: int = 2025,
    apenas_semestrais: bool = True,
) -> tuple[list, list]:
    selecionados = []
    ignorados = []
    vistos = set()
    for link in links:
        nome = link.get("nome", "")
        href = link.get("url", "")
        txt = _normalizar_texto(f"{nome} {href}")
        nome_norm = _normalizar_texto(nome)

        ano_match = re.search(r"(20\d{2})", txt)
        ano = int(ano_match.group(1)) if ano_match else None
        eh_semestre = "semestre" in txt

        if apenas_semestrais and not eh_semestre:
            ignorados.append(link)
            continue
        if ano is None or ano < ano_min or ano > ano_max:
            ignorados.append(link)
            continue
        if nome_norm in vistos:
            ignorados.append(link)
            continue
        vistos.add(nome_norm)
        selecionados.append(link)

    return selecionados, ignorados


def download_csv(url: str, destino: Path, chunk_size: int = 8192) -> Path:
    logger.info(f"Baixando: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(destino, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=destino.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Salvo em: {destino}")
    return destino


def download_todos_csvs(
    destino_dir: Path = DATA_RAW,
    ano_min: int = 2004,
    ano_max: int = 2025,
    apenas_semestrais: bool = True,
) -> list:
    destino_dir.mkdir(parents=True, exist_ok=True)
    links = descobrir_links_csv()
    links_total = len(links)
    ignorados = []

    if apenas_semestrais:
        links, ignorados = filtrar_links_csv(
            links,
            ano_min=ano_min,
            ano_max=ano_max,
            apenas_semestrais=True,
        )

    logger.info(
        f"Links encontrados: {links_total} | "
        f"considerados: {len(links)} | ignorados: {len(ignorados)}"
    )
    arquivos = []
    baixados = 0
    pulados = 0

    for link in links:
        nome_arquivo = re.sub(r"[^\w\-.]", "_", link["nome"])
        if not nome_arquivo.endswith(".csv"):
            nome_arquivo += ".csv"
        destino = destino_dir / nome_arquivo

        if destino.exists():
            logger.info(f"Arquivo já existe, pulando: {destino.name}")
            arquivos.append(destino)
            pulados += 1
            continue

        try:
            arquivo = download_csv(link["url"], destino)
            arquivos.append(arquivo)
            baixados += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar {link['url']}: {e}")

    logger.info(
        f"Download concluído: {len(arquivos)} arquivos "
        f"(baixados={baixados}, pulados={pulados}, ignorados={len(ignorados)})"
    )
    return arquivos


def _criar_cliente_s3():
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError(
            "Credenciais AWS não configuradas. Verifique o arquivo .env"
        )
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def upload_para_s3(
    arquivo_local: Path,
    s3_key: Optional[str] = None,
    bucket: str = S3_BUCKET,
) -> str:
    s3 = _criar_cliente_s3()

    if s3_key is None:
        s3_key = f"raw/{arquivo_local.name}"

    logger.info(f"Enviando {arquivo_local.name} -> s3://{bucket}/{s3_key}")
    s3.upload_file(str(arquivo_local), bucket, s3_key)

    uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Upload concluído: {uri}")
    return uri


def upload_todos_para_s3(diretorio: Path = DATA_RAW) -> list:
    uris = []
    csvs = list(diretorio.glob("*.csv"))
    logger.info(f"Enviando {len(csvs)} arquivos para S3...")

    for csv_file in tqdm(csvs, desc="Upload S3"):
        try:
            uri = upload_para_s3(csv_file)
            uris.append(uri)
        except (ValueError, OSError, BotoCoreError, ClientError) as e:
            logger.error(f"Erro no upload de {csv_file.name}: {e}")

    return uris


def download_de_s3(
    s3_key: str,
    destino: Optional[Path] = None,
    bucket: str = S3_BUCKET,
) -> Path:
    s3 = _criar_cliente_s3()

    if destino is None:
        nome_arquivo = s3_key.split("/")[-1]
        destino = DATA_RAW / nome_arquivo

    destino.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baixando s3://{bucket}/{s3_key} -> {destino}")
    s3.download_file(bucket, s3_key, str(destino))
    logger.info(f"Download concluído: {destino}")
    return destino
