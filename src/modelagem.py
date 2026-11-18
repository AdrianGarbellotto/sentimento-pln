"""Construção das quatro configurações e avaliação sem vazamento de dados."""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline

from .preprocessamento import preprocessar_texto

CONFIGURACOES = {
    "A": {"stopwords": "mantidas", "remover_stopwords": False, "vetorizador": "BoW"},
    "B": {"stopwords": "mantidas", "remover_stopwords": False, "vetorizador": "TF-IDF"},
    "C": {"stopwords": "removidas", "remover_stopwords": True, "vetorizador": "BoW"},
    "D": {"stopwords": "removidas", "remover_stopwords": True, "vetorizador": "TF-IDF"},
}


def criar_pipeline(config_id):
    """Cria vetorizador + regressão logística dentro de um único Pipeline."""
    config = CONFIGURACOES[config_id]
    preprocessor = partial(
        preprocessar_texto,
        remover_stopwords=config["remover_stopwords"],
    )
    if config["vetorizador"] == "BoW":
        vectorizer = CountVectorizer(
            preprocessor=preprocessor,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
        )
    else:
        vectorizer = TfidfVectorizer(
            preprocessor=preprocessor,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
        )
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([
        ("vetorizador", vectorizer),
        ("classificador", classifier),
    ])


def comparar_configuracoes(X_treino, y_treino):
    """Calcula CV apenas no treino e retorna a tabela e a configuração escolhida."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    linhas = []
    pipelines = {}

    for config_id, config in CONFIGURACOES.items():
        pipeline = criar_pipeline(config_id)
        metricas = cross_validate(
            pipeline,
            X_treino,
            y_treino,
            cv=cv,
            scoring={"acc": "accuracy", "f1": make_scorer(f1_score, pos_label="positivo", zero_division=0)},
            return_train_score=False,
            n_jobs=None,
        )

        # Fit somente no conjunto de treino para medir o vocabulário aprendido.
        pipeline.fit(X_treino, y_treino)
        vocabulario = len(pipeline.named_steps["vetorizador"].get_feature_names_out())
        pipelines[config_id] = pipeline
        linhas.append({
            "config": config_id,
            "stopwords": config["stopwords"],
            "vetorizador": config["vetorizador"],
            "vocabulario": vocabulario,
            "acc_cv_media": float(np.mean(metricas["test_acc"])),
            "acc_cv_desvio": float(np.std(metricas["test_acc"], ddof=1)),
            "f1_cv_media": float(np.mean(metricas["test_f1"])),
            "f1_cv_desvio": float(np.std(metricas["test_f1"], ddof=1)),
        })

    tabela = pd.DataFrame(linhas)
    # Critério definido antes de consultar o teste: maior F1 médio na CV;
    # em empate, maior acurácia média na CV.
    tabela_ordenada = tabela.sort_values(
        ["f1_cv_media", "acc_cv_media", "config"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    vencedora = str(tabela_ordenada.iloc[0]["config"])
    return tabela, vencedora, pipelines
