# Classificação de Asteroides Binários com Machine Learning

Projeto de **Machine Learning aplicado à Astronomia**, desenvolvido para investigar a classificação de asteroides em dois grupos:

* **Classe 1:** asteroides binários;
* **Classe 0:** asteroides não binários (*single*).

O projeto utiliza dados fotométricos e geométricos do **Sloan Digital Sky Survey (SDSS)** e explora técnicas de análise estatística, redução de dimensionalidade e classificação supervisionada.

---

## 🎯 Objetivo

O objetivo é investigar se características observacionais de asteroides podem ser utilizadas para **identificar automaticamente candidatos a sistemas binários**.

A motivação está no fato de que sistemas binários podem fornecer informações importantes sobre propriedades físicas dos asteroides, como massa, densidade, formação e evolução.

Em grandes levantamentos astronômicos, como o **Vera C. Rubin Observatory / LSST**, o volume de dados torna inviável uma análise exclusivamente manual. Métodos de Machine Learning podem atuar como uma ferramenta de **triagem e identificação de candidatos** para análises astronômicas posteriores.

> **Importante:** o classificador não constitui uma confirmação física de que um asteroide é binário. Ele busca identificar padrões nos dados que podem ser utilizados para selecionar candidatos.

---

## 🔭 Dados

Os dados utilizados neste projeto são provenientes do **Sloan Digital Sky Survey (SDSS)**, obtidos através do serviço **VizieR/TAP**.

Tabela utilizada:

```text
J/A+A/652/A59/sso
```

Cada registro corresponde a uma observação de um asteroide.

Um mesmo asteroide pode possuir **múltiplas observações**, sendo identificado pela coluna:

```text
Number
```

Após o tratamento dos dados, foram utilizadas **844 observações**.

---

## 📊 Variáveis utilizadas

O modelo utiliza oito características como variáveis preditoras:

| Variável    | Descrição               |
| ----------- | ----------------------- |
| `Geodist`   | Distância geocêntrica   |
| `Heliodist` | Distância heliocêntrica |
| `alpha`     | Ângulo de fase          |
| `umag`      | Magnitude na banda u    |
| `gmag`      | Magnitude na banda g    |
| `rmag`      | Magnitude na banda r    |
| `imag`      | Magnitude na banda i    |
| `zmag`      | Magnitude na banda z    |

A variável alvo é:

```text
classe
```

com:

```text
0 → não binário
1 → binário
```

---

# 🔎 Análise exploratória

Antes da construção do classificador, foi realizada uma análise exploratória das relações entre as variáveis.

## Matriz de correlação

A matriz de correlação revelou algumas relações importantes:

* `Geodist` e `Heliodist` apresentam correlação extremamente alta (~0,9996);
* as diferentes bandas fotométricas apresentam correlações elevadas;
* `alpha` apresenta correlação relativamente baixa com as magnitudes;
* as distâncias apresentam correlações moderadas com as magnitudes.

Essas relações indicam a existência de **redundância e estrutura compartilhada entre as variáveis**.

---

# 📐 PCA — Principal Component Analysis

Foi aplicada **Análise de Componentes Principais (PCA)** como ferramenta exploratória.

Antes da PCA, as variáveis foram padronizadas utilizando `StandardScaler`, garantindo que as diferenças de escala entre as variáveis não dominassem a análise.

Os primeiros componentes apresentaram a seguinte variância explicada:

| Componente | Variância explicada |
| ---------- | ------------------: |
| PC1        |              67,28% |
| PC2        |              21,51% |
| PC3        |               7,67% |
| PC4        |               2,17% |
| PC5        |               0,62% |
| PC6        |               0,41% |
| PC7        |               0,33% |
| PC8        |              0,003% |

Os três primeiros componentes explicam aproximadamente:

```text
96,47%
```

da variância total dos dados.

A PCA foi utilizada **sem fornecer a variável `classe`**, portanto os componentes principais não foram construídos para separar asteroides binários e não binários.

A identificação das classes nos gráficos de PCA foi realizada posteriormente apenas para **visualização exploratória**.

---

# 🤖 Classificação por Regressão Logística

Como primeiro modelo supervisionado, foi utilizada a **Regressão Logística** (`LogisticRegression` do Scikit-Learn).

O modelo calcula uma combinação linear das características:

$$
z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_8x_8
$$

e transforma esse valor em uma probabilidade através da função logística:

$$
P(\text{classe}=1)=\frac{1}{1+e^{-z}}
$$

Foi utilizado o limiar padrão de 0,5:

```text
P(classe = 1) ≥ 0,5 → binário
P(classe = 1) < 0,5 → não binário
```

---

# ⚠️ Prevenção de Data Leakage

Um dos principais cuidados metodológicos deste projeto foi evitar **vazamento de dados (data leakage)**.

Como o conjunto contém múltiplas observações do mesmo asteroide, simplesmente dividir as linhas aleatoriamente poderia colocar:

```text
Observação do asteroide X → treino
Outra observação do asteroide X → teste
```

Isso faria com que o modelo fosse testado em um objeto que já havia aparecido no treinamento.

Para evitar esse problema, foi utilizado:

```python
StratifiedGroupKFold
```

com:

```python
n_splits=5
shuffle=True
random_state=42
```

O agrupamento foi realizado pela coluna:

```python
groups = novo_df['Number']
```

Assim, todas as observações pertencentes ao mesmo asteroide permanecem no mesmo conjunto.

Além disso, a estratificação busca preservar aproximadamente a proporção entre as classes em cada partição.

---

# 🔬 Padronização

A padronização foi realizada **dentro de cada partição da validação cruzada**.

O `StandardScaler` é ajustado somente nos dados de treinamento:

```python
scaler.fit_transform(X_train)
```

e posteriormente aplicado aos dados de teste:

```python
scaler.transform(X_test)
```

Isso evita que informações estatísticas do conjunto de teste sejam utilizadas durante o treinamento.

---

# 🔄 Validação cruzada

Foi utilizada validação cruzada estratificada por grupos com cinco partições:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Em cada rodada:

1. Os asteroides são separados em treinamento e teste;
2. Não há sobreposição de `Number` entre os conjuntos;
3. As variáveis são padronizadas;
4. A regressão logística é treinada;
5. O modelo realiza as previsões no conjunto de teste;
6. As métricas são calculadas.

---

# 📈 Resultados

Os resultados obtidos nas cinco partições foram:

| Métrica   | Média ± Desvio-padrão |
| --------- | --------------------: |
| Accuracy  |     **87,49 ± 3,26%** |
| Precision |     **85,93 ± 5,11%** |
| Recall    |     **88,60 ± 6,57%** |
| F1-score  |     **87,00 ± 3,37%** |

Os valores representam a média e o desvio-padrão das cinco partições da validação cruzada.

### Accuracy

Proporção total de classificações corretas.

### Precision

Entre os objetos classificados como binários, indica a proporção que realmente pertence à classe binária.

### Recall

Entre os asteroides realmente binários, indica a proporção identificada pelo modelo.

### F1-score

Combina Precision e Recall através da média harmônica:

$$
F1 = 2\frac{Precision \times Recall}
{Precision + Recall}
$$

---

# 🧩 Matriz de confusão

As matrizes de confusão obtidas foram:

```text
Fold 1
[[88, 10],
 [10, 64]]

Fold 2
[[87, 12],
 [ 7, 68]]

Fold 3
[[57,  4],
 [14, 82]]

Fold 4
[[81, 18],
 [19, 78]]

Fold 5
[[69, 14],
 [ 0, 62]]
```

A matriz permite distinguir diferentes tipos de erro:

```text
TN → não binário corretamente classificado
TP → binário corretamente classificado
FP → não binário classificado como binário
FN → binário classificado como não binário
```

---

# 🧠 Interpretação dos coeficientes

A regressão logística também permite analisar os coeficientes associados às variáveis.

Coeficientes obtidos no modelo analisado:

| Feature     | Coeficiente |
| ----------- | ----------: |
| `Geodist`   |      +0,732 |
| `Heliodist` |      −0,752 |
| `alpha`     |      −0,186 |
| `umag`      |      −2,281 |
| `gmag`      |      +0,214 |
| `rmag`      |      +0,024 |
| `imag`      |      −0,370 |
| `zmag`      |      −0,039 |

O sinal do coeficiente indica a direção da contribuição para o valor `z` da regressão logística.

Entretanto, esses coeficientes **não devem ser interpretados diretamente como importância física das variáveis**, especialmente porque algumas características apresentam correlações muito fortes entre si.

---

# 🛠️ Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* PyVO
* VizieR/TAP
* Sloan Digital Sky Survey (SDSS)

---

# 📁 Estrutura prevista

```text
asteroid-ml/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── asteroid_classification.ipynb
│
├── src/
│   └── ...
│
├── figures/
│   ├── correlation_matrix.png
│   ├── pca_pc1_pc2.png
│   ├── pca_pc1_pc3.png
│   ├── confusion_matrix.png
│   └── model_metrics.png
│
├── requirements.txt
└── README.md
```

---

# 🚀 Próximos passos

O projeto ainda está em desenvolvimento. Entre as próximas etapas estão:

* análise mais aprofundada dos coeficientes;
* avaliação de outros algoritmos de Machine Learning;
* comparação entre modelos;
* análise de importância das variáveis;
* avaliação de possíveis efeitos de redundância entre features;
* investigação de curvas de luz e características temporais;
* análise de possíveis curvas de luz anômalas;
* desenvolvimento de modelos mais adequados para grandes levantamentos astronômicos.

---

# 📚 Motivação científica

A ideia central do projeto é investigar como técnicas de **Machine Learning podem auxiliar a Astronomia observacional na identificação de candidatos a asteroides binários**.

O SDSS fornece um conjunto de dados adequado para desenvolver e testar a metodologia em uma escala manejável. Posteriormente, técnicas semelhantes podem ser aplicadas a levantamentos de grande escala e domínio temporal, como o **Vera C. Rubin Observatory / LSST**.

Nesse contexto, o Machine Learning não substitui a análise astronômica: ele funciona como uma ferramenta para **encontrar padrões, reduzir o espaço de candidatos e direcionar análises posteriores**.

---

## 👨‍🔬 Autor

**Alan Coutinho**

Projeto desenvolvido como estudo de aplicação de Machine Learning à Astronomia, com foco na classificação de asteroides binários.

---

## ⚖️ Status

**Em desenvolvimento 🚧**

O modelo apresentado representa uma primeira etapa do projeto. Novos modelos, variáveis observacionais e análises serão incorporados nas próximas versões.
