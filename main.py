"""Executa toda a análise de sentimento e gera os artefatos em resultados/."""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.analise import explorar_dados, avaliar_modelo, avaliar_casos_dificeis
from src.modelagem import comparar_configuracoes, criar_pipeline

RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / "data"
PASTA_RESULTADOS = RAIZ / "resultados"


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    caminho_avaliacoes = PASTA_DADOS / "avaliacoes.csv"
    caminho_dificeis = PASTA_DADOS / "casos_dificeis.csv"
    if not caminho_avaliacoes.exists() or not caminho_dificeis.exists():
        raise FileNotFoundError(
            "Arquivos não encontrados. Confira data/avaliacoes.csv e data/casos_dificeis.csv."
        )

    avaliacoes = pd.read_csv(caminho_avaliacoes)
    casos_dificeis = pd.read_csv(caminho_dificeis)
    obrigatorias = {"id", "texto", "sentimento"}
    if not obrigatorias.issubset(avaliacoes.columns):
        raise ValueError(f"avaliacoes.csv deve conter as colunas: {sorted(obrigatorias)}")
    obrigatorias_casos = {"id", "texto", "sentimento_esperado", "tipo"}
    if not obrigatorias_casos.issubset(casos_dificeis.columns):
        raise ValueError(f"casos_dificeis.csv deve conter: {sorted(obrigatorias_casos)}")

    avaliacoes = avaliacoes.dropna(subset=["texto", "sentimento"]).copy()
    avaliacoes["texto"] = avaliacoes["texto"].astype(str)
    avaliacoes["sentimento"] = avaliacoes["sentimento"].astype(str).str.strip().str.lower()
    if set(avaliacoes["sentimento"].unique()) != {"positivo", "negativo"}:
        raise ValueError("O conjunto de treino deve conter somente positivo e negativo.")

    explorar_dados(avaliacoes, PASTA_RESULTADOS)

    X = avaliacoes["texto"]
    y = avaliacoes["sentimento"]
    indices_treino, indices_teste = train_test_split(
        avaliacoes.index,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )
    X_treino = X.loc[indices_treino]
    X_teste = X.loc[indices_teste]
    y_treino = y.loc[indices_treino]
    y_teste = y.loc[indices_teste]
    df_teste = avaliacoes.loc[indices_teste].reset_index(drop=True)

    tabela, vencedora, _ = comparar_configuracoes(X_treino, y_treino)
    tabela.to_csv(PASTA_RESULTADOS / "comparacao.csv", index=False, encoding="utf-8-sig")

    modelo_final = criar_pipeline(vencedora)
    modelo_final.fit(X_treino, y_treino)
    metricas_teste = avaliar_modelo(
        modelo_final, X_teste, y_teste, df_teste, PASTA_RESULTADOS
    )
    resumo_casos, predicoes_casos = avaliar_casos_dificeis(
        modelo_final, casos_dificeis, PASTA_RESULTADOS
    )

    config_vencedora = tabela.loc[tabela["config"] == vencedora].iloc[0].to_dict()
    resumo = {
        "configuracao_escolhida": vencedora,
        "criterio_escolha": "maior F1 médio na validação cruzada; desempate por acurácia média",
        "configuracao": {k: (float(v) if hasattr(v, "item") and isinstance(v, (int, float)) else v)
                         for k, v in config_vencedora.items()},
        "metricas_teste": metricas_teste,
        "casos_dificeis": resumo_casos,
        "observacao": "Casos mistos/neutros foram previstos como positivo ou negativo, mas não entraram no cálculo de acurácia dos casos difíceis.",
    }
    (PASTA_RESULTADOS / "resumo_execucao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Análise concluída.")
    print(f"Configuração escolhida pela validação cruzada: {vencedora}")
    print(f"Acurácia no teste: {metricas_teste['acuracia_teste']:.4f}")
    print(f"F1 positivo no teste: {metricas_teste['f1_positivo_teste']:.4f}")
    print(f"Resultados salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
