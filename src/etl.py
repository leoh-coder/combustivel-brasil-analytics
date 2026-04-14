import re
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from src.utils import (
    DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL,
    PRODUTOS_MAP, ESTADO_PARA_SIGLA, setup_logger,
)

logger = setup_logger("etl")


def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().upper()


def _normalizar_serie(serie: pd.Series) -> pd.Series:
    resultado = serie.astype(str).str.normalize("NFKD")
    resultado = resultado.str.encode("ascii", errors="ignore").str.decode("ascii")
    return resultado.str.strip().str.upper()


def _ler_csv(csv_file: Path) -> Optional[pd.DataFrame]:
    for sep, enc in [(";", "utf-8"), (";", "latin-1"), ("\t", "utf-8")]:
        try:
            return pd.read_csv(csv_file, sep=sep, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    return None


def carregar_csvs_anp(diretorio: Path = DATA_RAW) -> pd.DataFrame:
    csvs = sorted(diretorio.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV em {diretorio}")
    logger.info(f"Carregando {len(csvs)} CSVs...")
    dfs = []
    for csv_file in csvs:
        df = _ler_csv(csv_file)
        if df is None:
            logger.error(f"  Falha ao ler {csv_file.name}")
            continue
        dfs.append(df)
        logger.info(f"  {csv_file.name}: {len(df)} registros")

    if not dfs:
        logger.warning("Nenhum CSV foi lido com sucesso")
        return pd.DataFrame()

    df_total = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total: {len(df_total)} registros")
    return df_total


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        novo = normalizar_texto(col).lower()
        novo = re.sub(r"[^a-z0-9]+", "_", novo).strip("_")
        rename_map[col] = novo
    df = df.rename(columns=rename_map)

    alias = {
        "regiao_sigla": "regiao", "estado_sigla": "estado",
        "cnpj_da_revenda": "cnpj", "nome_da_rua": "rua",
        "numero_rua": "numero", "data_da_coleta": "data_coleta",
        "valor_de_venda": "valor_venda", "valor_de_compra": "valor_compra",
        "unidade_de_medida": "unidade",
    }
    alias_seguro = {
        k: v for k, v in alias.items()
        if k in df.columns and v not in df.columns
    }
    return df.rename(columns=alias_seguro)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Limpando dados...")
    n_antes = len(df)

    if "data_coleta" in df.columns:
        df["data_coleta"] = pd.to_datetime(df["data_coleta"], format="%d/%m/%Y", errors="coerce")

    for col in ["valor_venda", "valor_compra"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    df = df.dropna(subset=["valor_venda", "data_coleta"])
    df = df.query("0 < valor_venda < 50")
    logger.info(f"  Removidos {n_antes - len(df)} registros inválidos")

    if "produto" in df.columns:
        df["produto"] = _normalizar_serie(df["produto"])
        df["produto_padronizado"] = df["produto"].map(PRODUTOS_MAP).fillna(df["produto"])

    if "produto_padronizado" in df.columns:
        n_glp = (df["produto_padronizado"] == "GLP").sum()
        df = df[df["produto_padronizado"] != "GLP"]
        if n_glp:
            logger.info(f"  GLP removido (fora do escopo): {n_glp} registros")

    if "estado" in df.columns:
        df["estado"] = _normalizar_serie(df["estado"])
        df["estado_sigla"] = df["estado"].map(ESTADO_PARA_SIGLA).fillna(df["estado"])

    if "municipio" in df.columns:
        df["municipio"] = _normalizar_serie(df["municipio"])

    if "cnpj" in df.columns:
        df["cnpj"] = df["cnpj"].astype(str)

    if "bandeira" in df.columns:
        df["bandeira"] = _normalizar_serie(df["bandeira"])
        df.loc[df["bandeira"].str.contains("BRANCA", na=False), "bandeira"] = "BRANCA"

    if "data_coleta" in df.columns:
        df["ano"] = df["data_coleta"].dt.year
        df["mes"] = df["data_coleta"].dt.month
        df["ano_mes"] = df["data_coleta"].dt.to_period("M")

    dup_cols = None
    if "cnpj" in df.columns:
        dup_cols = [col for col in ["data_coleta", "cnpj", "produto"] if col in df.columns]
    n_dup = df.duplicated(subset=dup_cols).sum()
    if n_dup:
        df = df.drop_duplicates(subset=dup_cols)
        logger.info(f"  Removidas {n_dup} duplicatas")

    logger.info(f"  Resultado: {len(df)} registros limpos")
    return df


def enriquecer_com_dados_externos(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Enriquecendo com câmbio e petróleo...")
    df = df.sort_values("data_coleta")

    path_dolar = DATA_EXTERNAL / "cotacao_dolar.parquet"
    if path_dolar.exists():
        df_dolar = pd.read_parquet(path_dolar)
        df_dolar["data"] = pd.to_datetime(df_dolar["data"])
        df = pd.merge_asof(
            df,
            df_dolar[["data", "cotacao_venda"]]
                .sort_values("data")
                .rename(columns={"data": "data_coleta", "cotacao_venda": "dolar_venda"}),
            on="data_coleta", direction="backward",
        )
        logger.info("  Dólar: OK")

    path_brent = DATA_EXTERNAL / "preco_brent.parquet"
    if path_brent.exists():
        df_brent = pd.read_parquet(path_brent)
        df_brent["data"] = pd.to_datetime(df_brent["data"])
        df = pd.merge_asof(
            df,
            df_brent.sort_values("data").rename(columns={"data": "data_coleta"}),
            on="data_coleta", direction="backward",
        )
        logger.info("  Brent: OK")

    return df


def criar_aggregacoes(df: pd.DataFrame) -> dict:
    logger.info("Criando agregações...")
    aggs = {}

    aggs["mensal_estado_produto"] = (
        df.groupby(["ano", "mes", "estado_sigla", "produto_padronizado"])
        .agg(preco_medio=("valor_venda", "mean"), preco_mediano=("valor_venda", "median"),
             preco_min=("valor_venda", "min"), preco_max=("valor_venda", "max"),
             n_registros=("valor_venda", "count"))
        .reset_index()
    )

    aggs["mensal_nacional"] = (
        df.groupby(["ano", "mes", "produto_padronizado"])
        .agg(preco_medio=("valor_venda", "mean"), preco_mediano=("valor_venda", "median"),
             n_registros=("valor_venda", "count"))
        .reset_index()
    )

    aggs["por_bandeira"] = (
        df.groupby(["ano", "mes", "bandeira", "produto_padronizado"])
        .agg(preco_medio=("valor_venda", "mean"), n_registros=("valor_venda", "count"))
        .reset_index()
    )

    mask_pp = df["municipio"].str.contains("PRESIDENTE PRUDENTE", na=False)
    if mask_pp.any():
        aggs["presidente_prudente"] = (
            df[mask_pp]
            .groupby(["ano", "mes", "produto_padronizado", "bandeira"])
            .agg(preco_medio=("valor_venda", "mean"), n_registros=("valor_venda", "count"))
            .reset_index()
        )

    return aggs


def _criar_aggregacoes_duckdb(parquet_path: Path) -> dict:
    logger.info("Criando agregações...")
    p = str(parquet_path).replace("\\", "/")
    aggs = {}

    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{p}')")

        aggs["mensal_estado_produto"] = con.execute("""
            SELECT ano, mes, estado_sigla, produto_padronizado,
                   AVG(valor_venda) AS preco_medio, MEDIAN(valor_venda) AS preco_mediano,
                   MIN(valor_venda) AS preco_min, MAX(valor_venda) AS preco_max,
                   COUNT(*) AS n_registros
            FROM df
            GROUP BY ano, mes, estado_sigla, produto_padronizado
        """).fetchdf()

        aggs["mensal_nacional"] = con.execute("""
            SELECT ano, mes, produto_padronizado,
                   AVG(valor_venda) AS preco_medio, MEDIAN(valor_venda) AS preco_mediano,
                   COUNT(*) AS n_registros
            FROM df
            GROUP BY ano, mes, produto_padronizado
        """).fetchdf()

        aggs["por_bandeira"] = con.execute("""
            SELECT ano, mes, bandeira, produto_padronizado,
                   AVG(valor_venda) AS preco_medio, COUNT(*) AS n_registros
            FROM df
            GROUP BY ano, mes, bandeira, produto_padronizado
        """).fetchdf()

        aggs["presidente_prudente"] = con.execute("""
            SELECT ano, mes, produto_padronizado, bandeira,
                   AVG(valor_venda) AS preco_medio, COUNT(*) AS n_registros
            FROM df
            WHERE municipio LIKE '%PRESIDENTE PRUDENTE%'
            GROUP BY ano, mes, produto_padronizado, bandeira
        """).fetchdf()

    return aggs


def executar_queries_duckdb(parquet_path: Path) -> dict:
    logger.info("Executando queries DuckDB...")
    resultados = {}

    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW combustiveis AS SELECT * FROM read_parquet('{parquet_path}')"
        )

        resultados["top_estados_gasolina"] = con.execute("""
            SELECT estado_sigla, AVG(valor_venda) as preco_medio, COUNT(*) as n_registros
            FROM combustiveis
            WHERE produto_padronizado = 'Gasolina Comum'
            GROUP BY estado_sigla
            ORDER BY preco_medio DESC
            LIMIT 10
        """).fetchdf()

        resultados["evolucao_anual"] = con.execute("""
            SELECT ano, produto_padronizado,
                   AVG(valor_venda) as preco_medio,
                   MEDIAN(valor_venda) as preco_mediano,
                   COUNT(*) as n_registros
            FROM combustiveis
            GROUP BY ano, produto_padronizado
            ORDER BY ano, produto_padronizado
        """).fetchdf()

        resultados["spread_bandeiras"] = con.execute("""
            SELECT bandeira, produto_padronizado,
                   AVG(valor_venda) as preco_medio, COUNT(*) as n_registros
            FROM combustiveis
            WHERE bandeira IS NOT NULL
            GROUP BY bandeira, produto_padronizado
            HAVING COUNT(*) > 1000
            ORDER BY produto_padronizado, preco_medio DESC
        """).fetchdf()

        resultados["pp_vs_sp_vs_brasil"] = con.execute("""
            SELECT ano, mes, produto_padronizado,
                AVG(CASE WHEN municipio LIKE '%PRESIDENTE PRUDENTE%' THEN valor_venda END) as preco_pp,
                AVG(CASE WHEN estado_sigla = 'SP' THEN valor_venda END) as preco_sp,
                AVG(valor_venda) as preco_brasil
            FROM combustiveis
            WHERE produto_padronizado = 'Gasolina Comum'
            GROUP BY ano, mes, produto_padronizado
            ORDER BY ano, mes
        """).fetchdf()

    return resultados


def executar_pipeline_etl(salvar_parquet: bool = True, tamanho_lote: int = 10) -> None:
    logger.info("Iniciando pipeline ETL...")

    csvs = sorted(DATA_RAW.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV em {DATA_RAW}")

    tmp_dir = DATA_PROCESSED / "_tmp_etl"
    tmp_dir.mkdir(exist_ok=True)

    logger.info(f"Processando {len(csvs)} CSVs em lotes de {tamanho_lote}...")
    for i in range(0, len(csvs), tamanho_lote):
        lote_csvs = csvs[i: i + tamanho_lote]
        dfs = []
        for csv_file in lote_csvs:
            df_csv = _ler_csv(csv_file)
            if df_csv is None:
                logger.error(f"  Falha ao ler {csv_file.name} — ignorado")
                continue
            dfs.append(df_csv)

        if not dfs:
            continue

        df_lote = pd.concat(dfs, ignore_index=True)
        del dfs
        df_lote = padronizar_colunas(df_lote)
        df_lote = limpar_dados(df_lote)
        df_lote = enriquecer_com_dados_externos(df_lote)

        num = i // tamanho_lote + 1
        tmp_path = tmp_dir / f"lote_{num:04d}.parquet"
        df_lote.to_parquet(tmp_path, index=False)
        logger.info(f"Lote {num}: {len(df_lote)} registros limpos")
        del df_lote

    logger.info("Combinando lotes via DuckDB...")
    glob_pattern = str(tmp_dir / "lote_*.parquet").replace("\\", "/")
    path = DATA_PROCESSED / "combustiveis_brasil.parquet"
    path_str = str(path).replace("\\", "/")

    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW lotes_etl AS SELECT * FROM read_parquet('{glob_pattern}')")

        n_total = con.execute("SELECT COUNT(*) FROM lotes_etl").fetchone()[0]
        logger.info(f"Total consolidado (bruto): {n_total} registros")

        colunas = {row[0] for row in con.execute("DESCRIBE lotes_etl").fetchall()}
        chave_preferencial = ["data_coleta", "cnpj", "produto_padronizado"]
        chave_dedup = [col for col in chave_preferencial if col in colunas]

        if chave_dedup:
            particao = ", ".join(chave_dedup)
            ordenacao = ", ".join(chave_dedup)
            n_final = con.execute(f"""
                SELECT COUNT(*)
                FROM (
                    SELECT ROW_NUMBER() OVER (
                        PARTITION BY {particao}
                        ORDER BY {ordenacao}
                    ) AS rn
                    FROM lotes_etl
                )
                WHERE rn = 1
            """).fetchone()[0]
            logger.info(
                f"Total após deduplicação global ({', '.join(chave_dedup)}): {n_final} registros"
            )
            con.execute(f"""
                COPY (
                    SELECT * EXCLUDE (rn)
                    FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY {particao}
                                   ORDER BY {ordenacao}
                               ) AS rn
                        FROM lotes_etl
                    )
                    WHERE rn = 1
                ) TO '{path_str}' (FORMAT PARQUET)
            """)
        else:
            logger.warning(
                "Deduplicação global ignorada: nenhuma coluna da chave preferencial "
                "está disponível no schema consolidado."
            )
            con.execute(
                f"COPY (SELECT * FROM lotes_etl) TO '{path_str}' (FORMAT PARQUET)"
            )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Salvo: {path}")

    if salvar_parquet:
        for nome, agg_df in _criar_aggregacoes_duckdb(path).items():
            agg_path = DATA_PROCESSED / f"agg_{nome}.parquet"
            agg_df.to_parquet(agg_path, index=False)

    logger.info("ETL concluído")
