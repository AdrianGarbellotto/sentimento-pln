# RELATÓRIO — Análise de Sentimento com PLN

**Nome:** Adrian dos Santos Garbellotto  
**Repositório:** [preencher com a URL do seu repositório no GitHub]  
**Data:** 17/11/2026

## Tabela-resumo

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---:|---:|---:|
| A | mantidas | BoW | 425 | 0,8410 ± 0,0799 | 0,8401 ± 0,0845 |
| B | mantidas | TF-IDF | 425 | 0,8205 ± 0,1026 | 0,8189 ± 0,1006 |
| C | removidas | BoW | 395 | 0,8462 ± 0,0850 | 0,8478 ± 0,0854 |
| D | removidas | TF-IDF | 395 | 0,8410 ± 0,0949 | 0,8398 ± 0,1052 |

**Configuração escolhida:** C — BoW com remoção de stopwords (preservando negações).  
**Acurácia no teste:** 0,8462. **F1 da classe positiva no teste:** 0,8529.

---

## 1. Pré-processamento

Primeiro, converti os textos para letras minúsculas e removi números e pontuação.  
A expressão regular manteve letras acentuadas do português, e a tokenização separou o texto em palavras.  
Montei uma lista própria com 65 stopwords comuns para testar a remoção na configuração C e D.  
A palavra “não” foi mantida, assim como “nunca” e “sem”, porque são importantes para o sentido da frase.  
Por exemplo, retirar “não” de “não gostei” poderia transformar uma avaliação negativa em uma leitura positiva.  
O vocabulário aprendido no treino ficou com 425 termos sem remoção e 395 com remoção de stopwords.

## 2. BoW vs. TF-IDF

A configuração C teve o maior F1 médio na validação cruzada: 0,8478, com desvio de 0,0854.  
A configuração A ficou próxima, com F1 de 0,8401 ± 0,0845, enquanto B teve 0,8189 ± 0,1006.  
A configuração D alcançou 0,8398 ± 0,1052; os desvios mostram que existe variação entre os folds e que as diferenças não são enormes.  
Escolhi C pelo maior F1 médio na validação cruzada, sem usar o conjunto de teste para decidir.  
BoW representa a quantidade de ocorrências das palavras, enquanto TF-IDF também considera o quanto cada termo é comum ou raro nos documentos.  
Neste conjunto, a contagem de palavras com stopwords removidas teve o melhor resultado médio de F1.

## 3. Avaliação no teste

A avaliação final foi feita uma única vez em 65 textos que ficaram separados do treino.  
A matriz de confusão, na ordem negativo e positivo, foi [[26, 7], [3, 29]], com acurácia de 0,8462.  
Para negativo, a precisão foi 0,8966, o recall 0,7879 e o F1 0,8387.  
Para positivo, a precisão foi 0,8056, o recall 0,9062 e o F1 0,8529.  
Neste contexto, classificar uma reclamação como elogio pode ser mais grave, pois a loja pode deixar de tratar um problema real do cliente.  
A acurácia é mais informativa aqui porque as classes estão equilibradas: são 130 avaliações positivas e 130 negativas.

## 4. Validação cruzada vs. divisão única

Separei 75% dos dados para treino e 25% para teste, com estratificação e `random_state=42`.  
Assim, o treino ficou com 195 avaliações e o teste final com 65.  
A validação cruzada estratificada de cinco folds compara cada configuração em diferentes partições do treino e mostra a variação entre elas.  
A escolha foi feita pelo F1 médio da validação cruzada, antes de consultar o teste, para não favorecer uma configuração por acaso.  
O vetorizador e a regressão ficaram juntos em um `Pipeline`, então o vocabulário é ajustado somente dentro dos dados de treino de cada fold.  
Isso evita o vazamento de dados que ocorreria ao aprender o vocabulário usando também os textos de validação ou de teste.

## 5. Análise de erros e palavras influentes

O modelo errou 10 dos 65 textos do teste.  
Um exemplo foi “O sabor é maravilhoso e as porções são generosas”, que era positivo, mas foi previsto como negativo.  
Também errou “Panela de péssima qualidade, a comida gruda e queima”, prevendo positivo apesar das expressões negativas.  
Em “Cheguei animado, mas me arrependi, tudo estava sujo”, o modelo deu peso ao início da frase e não captou bem a mudança de sentido.  
Entre os termos positivos apareceram “bastante” e “bem”, que não expressam sentimento positivo em todos os contextos; entre os negativos apareceram “ainda” e “toda”.  
Isso mostra que o modelo aprende associações com as palavras do conjunto, mas não entende o contexto completo como uma pessoa.

## 6. Casos difíceis e limitações

Nos 13 casos difíceis, 10 tinham rótulo binário e o modelo acertou 6, ou seja, 60% desses casos.  
Acertou exemplos como “Não gostei nada do atendimento” e “Não é ruim, gostei bastante”, mas falhou em algumas negações mais indiretas.  
Na ironia, confundiu “Que ótimo, o produto chegou quebrado” e “Maravilha, mais uma vez a entrega atrasou”, pois as palavras positivas aparecem em frases negativas.  
As duas frases mistas e a frase neutra também receberam uma das duas classes, porque o modelo só foi treinado para positivo e negativo; elas não entraram no cálculo de acurácia dos casos binários.  
Duas melhorias possíveis seriam testar bigramas (sequências de duas palavras) e criar atributos que preservem a relação entre a negação e a palavra seguinte.  
O conjunto é pequeno, balanceado e parcialmente gerado por padrões; em produção, diferenças de linguagem, ironia e assuntos novos poderiam reduzir bastante a confiabilidade.

---

## Declaração de uso de IA

- [x] Usei IA generativa como apoio na estruturação do projeto, implementação do código e organização/revisão deste relatório. Os resultados numéricos foram gerados pela execução de `python main.py` sobre os arquivos CSV fornecidos. Esta declaração deve ser mantida de acordo com a autorização e as orientações atualizadas do professor.
