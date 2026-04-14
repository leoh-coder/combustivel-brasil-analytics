import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl import executar_pipeline_etl, executar_queries_duckdb
from src.scraping import salvar_dados_externos
from src.utils import DATA_PROCESSED, setup_logger

logger = setup_logger("run_etl")


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline ETL completo")
    parser.add_argument(
        "--com-scraping",
        action="store_true",
        help="Coletar dados externos (dólar e Brent) antes do ETL",
    )
    parser.add_argument(
        "--com-duckdb",
        action="store_true",
        help="Executar queries analíticas com DuckDB após o ETL",
    )
    args = parser.parse_args()

    if args.com_scraping:
        logger.info("Coletando dados externos...")
        paths = salvar_dados_externos()
        for nome, path in paths.items():
            logger.info(f"  {nome}: {path}")

    executar_pipeline_etl()
    logger.info("Pipeline ETL concluído com sucesso")

    if args.com_duckdb:
        parquet_path = DATA_PROCESSED / "combustiveis_brasil.parquet"
        if parquet_path.exists():
            resultados = executar_queries_duckdb(parquet_path)
            for nome, result_df in resultados.items():
                logger.info(f"\n--- {nome} ---")
                logger.info(f"\n{result_df.head(10)}")
        else:
            logger.error(f"Arquivo Parquet não encontrado: {parquet_path}")


if __name__ == "__main__":
    main()
