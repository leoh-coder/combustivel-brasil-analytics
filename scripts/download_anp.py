import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestao import download_todos_csvs
from src.utils import DATA_RAW, setup_logger

logger = setup_logger("download_anp")


def main():
    parser = argparse.ArgumentParser(
        description="Download automático dos CSVs de preços de combustíveis da ANP"
    )
    parser.add_argument(
        "--destino",
        type=str,
        default=str(DATA_RAW),
        help="Diretório para salvar os CSVs (default: data/raw/)",
    )
    args = parser.parse_args()

    destino = Path(args.destino)
    logger.info(f"Destino: {destino}")

    arquivos = download_todos_csvs(destino)
    logger.info(f"Download concluído! {len(arquivos)} arquivos salvos em {destino}")


if __name__ == "__main__":
    main()
