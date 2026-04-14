from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

import src.etl as etl
from src.etl import normalizar_texto, padronizar_colunas, limpar_dados


class TestNormalizarTexto:
    def test_remove_acentos(self):
        assert normalizar_texto("São Paulo") == "SAO PAULO"
        assert normalizar_texto("Paraná") == "PARANA"
        assert normalizar_texto("Ceará") == "CEARA"

    def test_uppercase(self):
        assert normalizar_texto("gasolina comum") == "GASOLINA COMUM"

    def test_strip(self):
        assert normalizar_texto("  São Paulo  ") == "SAO PAULO"

    def test_non_string(self):
        assert normalizar_texto(123) == 123
        assert normalizar_texto(None) is None


class TestPadronizarColunas:
    def test_renomeia_colunas(self):
        df = pd.DataFrame({
            "Região - Sigla": ["SE"],
            "Estado - Sigla": ["SP"],
            "Município": ["SAO PAULO"],
        })
        resultado = padronizar_colunas(df)
        assert all(col == col.lower() for col in resultado.columns)
        assert all(" " not in col for col in resultado.columns)

    def test_padronizar_colunas_sem_colisao(self):
        df = pd.DataFrame({
            "Estado": ["São Paulo"],
            "Estado - Sigla": ["SP"],
            "Município": ["São Paulo"],
        })
        resultado = padronizar_colunas(df)
        assert len(resultado.columns) == len(set(resultado.columns))


class TestLimparDados:
    def test_converte_datas(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert pd.api.types.is_datetime64_any_dtype(df["data_coleta"])

    def test_converte_valores(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert pd.api.types.is_numeric_dtype(df["valor_venda"])

    def test_remove_valores_absurdos(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert (df["valor_venda"] < 50).all()
        assert (df["valor_venda"] > 0).all()

    def test_remove_nulos(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert df["data_coleta"].notna().all()
        assert df["valor_venda"].notna().all()

    def test_padroniza_produto(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert "produto_padronizado" in df.columns

    def test_cria_ano_mes(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert "ano" in df.columns
        assert "mes" in df.columns

    def test_estado_sigla(self, df_etl_base):
        df = limpar_dados(df_etl_base)
        assert "estado_sigla" in df.columns
        assert "SP" in df["estado_sigla"].values

    def test_dataframe_vazio(self):
        df_vazio = pd.DataFrame(
            columns=["data_coleta", "valor_venda", "produto", "estado", "municipio", "bandeira"]
        )
        resultado = limpar_dados(df_vazio)
        assert resultado.empty

    def test_todos_valores_nulos(self, df_etl_base):
        df = df_etl_base.copy()
        df["valor_venda"] = None
        resultado = limpar_dados(df)
        assert resultado.empty

    def test_produto_desconhecido(self, df_etl_base):
        df = df_etl_base.copy()
        df["produto"] = "QUEROSENE AVIACAO"
        resultado = limpar_dados(df)
        assert "produto_padronizado" in resultado.columns
        assert (resultado["produto_padronizado"] == "QUEROSENE AVIACAO").all()

    def test_deduplicacao_sem_produto_nao_quebra(self):
        df = pd.DataFrame({
            "data_coleta": ["01/01/2024", "01/01/2024"],
            "valor_venda": ["5,10", "5,10"],
            "cnpj": ["12345678000190", "12345678000190"],
            "estado": ["SAO PAULO", "SAO PAULO"],
            "municipio": ["PRESIDENTE PRUDENTE", "PRESIDENTE PRUDENTE"],
            "bandeira": ["BRANCA", "BRANCA"],
        })
        resultado = limpar_dados(df)
        assert len(resultado) == 1


class TestPipelineDeduplicacaoGlobal:
    def test_remove_duplicatas_entre_lotes_no_consolidado(self, monkeypatch):
        base = Path(f"etl_dedup_test_{uuid4().hex}")
        try:
            raw_dir = base / "raw"
            processed_dir = base / "processed"
            external_dir = base / "external"
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            external_dir.mkdir(parents=True)

            csv_content = (
                "Data da Coleta;CNPJ da Revenda;Produto;Valor de Venda\n"
                "01/01/2025;12345678000190;GASOLINA COMUM;5,10\n"
            )
            (raw_dir / "lote_a.csv").write_text(csv_content, encoding="utf-8")
            (raw_dir / "lote_b.csv").write_text(csv_content, encoding="utf-8")

            monkeypatch.setattr(etl, "DATA_RAW", Path(raw_dir))
            monkeypatch.setattr(etl, "DATA_PROCESSED", Path(processed_dir))
            monkeypatch.setattr(etl, "DATA_EXTERNAL", Path(external_dir))

            etl.executar_pipeline_etl(salvar_parquet=False, tamanho_lote=1)

            saida = pd.read_parquet(processed_dir / "combustiveis_brasil.parquet")
            chave = [
                col for col in ["data_coleta", "cnpj", "produto_padronizado"]
                if col in saida.columns
            ]
            duplicatas = saida.duplicated(subset=chave).sum() if chave else 0

            assert len(saida) == 1
            assert duplicatas == 0
        finally:
            shutil.rmtree(base, ignore_errors=True)
