# Análise de Sentimento com PLN

Projeto didático para classificar avaliações em português como positivas ou negativas, comparar Bag of Words e TF-IDF e avaliar o modelo sem vazamento de dados.

## Requisitos

- Python 3.10 ou superior
- pip

## Instalação (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente, abra um terminal CMD no VS Code e execute:

```cmd
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Execução

Na raiz do projeto, execute:

```bash
python main.py
```

O programa lê `data/avaliacoes.csv` e `data/casos_dificeis.csv` e gera os arquivos automaticamente em `resultados/`.

O projeto foi executado neste ambiente com Python 3.13, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0 e matplotlib 3.10.8. O `requirements.txt` usa intervalos de versões para permitir instalação em diferentes versões de Python 3.10 ou superior.

## Arquivos gerados

- `exploracao.txt`: volume, distribuição, comprimento médio e exemplos.
- `comparacao.csv`: média e desvio padrão de acurácia e F1 na validação cruzada.
- `matriz_confusao.csv` e `matriz_confusao.png`.
- `relatorio_classificacao.txt`: métricas de cada classe no teste.
- `erros.csv`: previsões incorretas e probabilidades.
- `palavras_influentes.csv`: 10 coeficientes mais positivos e 10 mais negativos.
- `casos_dificeis_predicoes.csv`: previsões dos 13 casos de teste adicionais.
- `resumo_execucao.json`: configuração escolhida e métricas resumidas.

## Metodologia

O conjunto é dividido uma única vez em 75% treino e 25% teste, com estratificação e `random_state=42`. As quatro configurações são comparadas somente no treino por validação cruzada estratificada de cinco folds. Vetorizador e classificador ficam juntos em um `Pipeline`, de modo que o vocabulário é aprendido dentro de cada fold. A escolha usa o maior F1 médio da validação cruzada; a acurácia média é usada como desempate. O conjunto de teste só é usado na avaliação final.

A palavra `não` é preservada mesmo na configuração com remoção de stopwords, pois sua retirada pode inverter o sentido de expressões como “não gostei”. O mesmo cuidado é aplicado a “nunca” e “sem”.

## Commits sugeridos

Faça os commits durante o desenvolvimento, em etapas reais, por exemplo:

1. `estrutura inicial e dados do projeto`
2. `adiciona exploração do conjunto de avaliações`
3. `implementa normalização e stopwords`
4. `compara BoW e TF-IDF com validação cruzada`
5. `adiciona avaliação final e análise de erros`
6. `documenta resultados e limitações no relatório`

A declaração de uso de IA no relatório deve refletir o uso que realmente ocorreu e as orientações atualizadas do professor.
