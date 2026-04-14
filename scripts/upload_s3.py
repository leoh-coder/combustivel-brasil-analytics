import argparse
import sys
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestao import upload_para_s3
from src.utils import DATA_RAW, DATA_PROCESSED, setup_logger

logger = setup_logger("upload_s3")


def upload_diretorio(diretorio: Path, prefixo_s3: str) -> list:
    arquivos = list(diretorio.glob("*.*"))
    if not arquivos:
        logger.warning(f"Nenhum arquivo encontrado em {diretorio}")
        return []

    uris = []
    for arquivo in arquivos:
        try:
            s3_key = f"{prefixo_s3}/{arquivo.name}"
            uri = upload_para_s3(arquivo, s3_key=s3_key)
            uris.append(uri)
        except (ValueError, OSError, BotoCoreError, ClientError) as e:
            logger.error(f"Erro no upload de {arquivo.name}: {e}")

    return uris


def main():
    parser = argparse.ArgumentParser(description="Upload de dados para AWS S3")
    parser.add_argument(
        "--tipo",
        choices=["raw", "processed", "all"],
        default="raw",
        help="Tipo de dados para upload (default: raw)",
    )
    args = parser.parse_args()

    if args.tipo in ("raw", "all"):
        logger.info("Enviando dados brutos (raw/)...")
        uris = upload_diretorio(DATA_RAW, "raw")
        logger.info(f"Raw: {len(uris)} arquivos enviados")

    if args.tipo in ("processed", "all"):
        logger.info("Enviando dados processados (processed/)...")
        uris = upload_diretorio(DATA_PROCESSED, "processed")
        logger.info(f"Processed: {len(uris)} arquivos enviados")


if __name__ == "__main__":
    main()
