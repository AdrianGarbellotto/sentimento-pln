"""Exploração, avaliação final e geração dos arquivos de resultados."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABELS = ["negativo", "positivo"]


def explorar_dados(df, pasta_resultados):
    linhas = []
    linhas.append("EXPLORAÇÃO DO DATASET DE AVALIAÇÕES")
    linhas.append(f"Total de avaliações: {len(df)}")
    linhas.append("Distribuição das classes:")
    contagem = df["sentimento"].value_counts().reindex(LABELS, fill_value=0)
    for classe, quantidade in contagem.items():
        linhas.append(f"  {classe}: {quantidade} ({quantidade / len(df) * 100:.1f}%)")
    linhas.append("")
    linhas.append("Tamanho médio dos textos (palavras), por classe:")
    tamanhos = df["texto"].fillna("").astype(str).str.split().str.len()
    for classe in LABELS:
        media = tamanhos[df["sentimento"] == classe].mean()
        linhas.append(f"  {classe}: {media:.2f}")
    linhas.append("")
    linhas.append("Três exemplos de cada classe:")
    for classe in LABELS:
        linhas.append(f"\n{classe.upper()}:")
        exemplos = df.loc[df["sentimento"] == classe, "texto"].head(3)
        for texto in exemplos:
            linhas.append(f"  - {texto}")
    (pasta_resultados / "exploracao.txt").write_text("\n".join(linhas), encoding="utf-8")


def avaliar_modelo(pipeline, X_teste, y_teste, df_teste, pasta_resultados):
    previsoes = pipeline.predict(X_teste)
    probabilidades = pipeline.predict_proba(X_teste)
    classes = list(pipeline.named_steps["classificador"].classes_)
    indice_positivo = classes.index("positivo")
    indice_negativo = classes.index("negativo")
    p_positivo = probabilidades[:, indice_positivo]

    acc = accuracy_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes, pos_label="positivo", zero_division=0)
    matriz = confusion_matrix(y_teste, previsoes, labels=LABELS)
    pd.DataFrame(matriz, index=["real_negativo", "real_positivo"],
                 columns=["previsto_negativo", "previsto_positivo"]).to_csv(
        pasta_resultados / "matriz_confusao.csv", encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    imagem = ax.imshow(matriz, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(imagem, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)), yticks=np.arange(len(LABELS)),
        xticklabels=LABELS, yticklabels=LABELS,
        xlabel="Sentimento previsto", ylabel="Sentimento real",
        title="Matriz de confusão — conjunto de teste",
    )
    limite = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > limite else "black")
    fig.tight_layout()
    fig.savefig(pasta_resultados / "matriz_confusao.png", dpi=160)
    plt.close(fig)

    relatorio = classification_report(
        y_teste, previsoes, labels=LABELS, target_names=LABELS,
        digits=4, zero_division=0,
    )
    (pasta_resultados / "relatorio_classificacao.txt").write_text(
        f"Acurácia no teste: {acc:.4f}\nF1 (positivo) no teste: {f1:.4f}\n\n{relatorio}",
        encoding="utf-8",
    )

    erros = []
    for pos, (real, previsto) in enumerate(zip(y_teste, previsoes)):
        if real != previsto:
            p_prevista = probabilidades[pos, classes.index(previsto)]
            erros.append({
                "texto": str(df_teste.iloc[pos]["texto"]),
                "real": real,
                "previsto": previsto,
                "probabilidade": float(p_prevista),
                "probabilidade_positivo": float(p_positivo[pos]),
            })
    pd.DataFrame(erros, columns=[
        "texto", "real", "previsto", "probabilidade", "probabilidade_positivo"
    ]).to_csv(pasta_resultados / "erros.csv", index=False, encoding="utf-8-sig")

    # Coeficientes positivos favorecem a classe positiva; negativos favorecem a negativa.
    vetor = pipeline.named_steps["vetorizador"]
    classificador = pipeline.named_steps["classificador"]
    termos = vetor.get_feature_names_out()
    coef = classificador.coef_[0]
    indice_top_pos = np.argsort(coef)[-10:][::-1]
    indice_top_neg = np.argsort(coef)[:10]
    influentes = []
    for idx in indice_top_pos:
        influentes.append({"palavra": termos[idx], "sentimento": "positivo", "coeficiente": float(coef[idx])})
    for idx in indice_top_neg:
        influentes.append({"palavra": termos[idx], "sentimento": "negativo", "coeficiente": float(coef[idx])})
    pd.DataFrame(influentes).to_csv(
        pasta_resultados / "palavras_influentes.csv", index=False, encoding="utf-8-sig"
    )

    metricas = {
        "acuracia_teste": float(acc),
        "f1_positivo_teste": float(f1),
        "total_teste": int(len(y_teste)),
        "total_erros": int(len(erros)),
        "matriz_confusao": matriz.tolist(),
    }
    return metricas


def avaliar_casos_dificeis(pipeline, df_casos, pasta_resultados):
    textos = df_casos["texto"].fillna("").astype(str)
    previsoes = pipeline.predict(textos)
    probabilidades = pipeline.predict_proba(textos)
    classes = list(pipeline.named_steps["classificador"].classes_)
    indice_positivo = classes.index("positivo")
    saida = pd.DataFrame({
        "texto": textos,
        "tipo": df_casos["tipo"],
        "esperado": df_casos["sentimento_esperado"],
        "previsto": previsoes,
        "probabilidade_positivo": probabilidades[:, indice_positivo],
    })
    saida.to_csv(
        pasta_resultados / "casos_dificeis_predicoes.csv",
        index=False, encoding="utf-8-sig",
    )
    validos = saida[saida["esperado"].isin(LABELS)]
    acertos = int((validos["esperado"] == validos["previsto"]).sum())
    resumo = {
        "total_casos": int(len(saida)),
        "casos_com_rotulo_binario": int(len(validos)),
        "acertos_nos_casos_binarios": acertos,
        "acuracia_nos_casos_binarios": float(acertos / len(validos)) if len(validos) else None,
        "casos_mistos_neutros_nao_avaliados_com_acuracia": int(len(saida) - len(validos)),
    }
    return resumo, saida
