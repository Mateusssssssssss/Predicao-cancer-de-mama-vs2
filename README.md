# Breast Cancer Classification

Projeto de classificação de tumores mamários benignos e malignos usando o Wisconsin Diagnostic Breast Cancer dataset. A reorganização preserva o XGBoost, o Random Forest, a escolha de threshold via precision-recall e as métricas/imagens já documentadas.

> **Uso educacional:** este modelo não é dispositivo médico e não deve ser usado para diagnóstico ou decisão clínica.

## Estrutura

```text
Predicao-cancer-de-mama-vs2/
├── data/
│   ├── raw/
│   │   ├── cancer_mama.csv
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── images/
│   ├── boxplot.png
│   ├── correlacao_variaveis.png
│   ├── curva_roc.png
│   ├── distribuicao_classes.png
│   └── distribuicao_dados.png
├── models/
│   ├── candidates/
│   │   ├── random_forest.joblib
│   │   └── xgboost.joblib
│   └── best_model/
│       └── xgboost_best.joblib
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_treinamento_e_avaliacao.ipynb
│   └── 03_cross_val_predict_oof.ipynb
├── reports/
│   ├── metrics.json
│   ├── oof/
│   │   └── oof_predictions.csv
│   ├── validation/
│   │   └── threshold_analysis.csv
│   └── figures/
│       ├── boxplot.png
│       ├── correlacao_variaveis.png
│       ├── curva_roc.png
│       ├── curva_pr.png
│       ├── distribuicao_classes.png
│       ├── distribuicao_dados.png
│       └── matriz_confusao.png
├── src/breast_cancer/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── validation.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── venv/
```

Os arquivos de dados, modelos, relatórios e o ambiente virtual são locais e ignorados pelo Git; os `.gitkeep` preservam as pastas vazias na estrutura versionada.

## Configuração

Requer Python 3.10 ou superior. Crie um ambiente e instale o projeto:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[notebooks]"
```

Coloque o CSV original em `data/raw/cancer_mama.csv`. O CSV deve conter `diagnosis` (`B` benigno, `M` maligno) e os atributos do dataset. O arquivo não foi incluído neste repositório; os dados locais são ignorados pelo Git.

## Execução

Abra os notebooks em ordem:

1. `notebooks/01_analise_exploratoria.ipynb`: validação do dataset, estatísticas, classes, correlações e figuras.
2. `notebooks/02_treinamento_e_avaliacao.ipynb`: split estratificado em treino, validação e teste; seleção do threshold na validação; avaliação final no teste.
3. `notebooks/03_cross_val_predict_oof.ipynb`: demonstração de `cross_val_predict` no treino e comparação OOF/in-sample.

Execute a partir da raiz do repositório. O segundo notebook compara candidatos por acurácia média de validação cruzada somente no treino, escolhe o threshold maximizando F2 na validação e avalia o vencedor uma única vez no teste. Salva candidatos em `models/candidates/`, o modelo final em `models/best_model/`, previsões OOF em `reports/oof/`, análise de threshold em `reports/validation/`, métricas em `reports/metrics.json` e gráficos em `reports/figures/`.

Os módulos têm responsabilidades separadas:

- `config.py`: caminhos, seed, proporção do split, parâmetros e β usado pelo F-beta.
- `data.py`: leitura do CSV e remoção de colunas de identificação/vazias.
- `preprocessing.py`: limpeza, separação entre `X` e `y`, codificação e split estratificado.
- `models.py`: fábricas para Random Forest, XGBoost e modelos candidatos.
- `validation.py`: validação cruzada, previsões OOF, busca em grade/aleatória e seleção/análise do threshold. Também expõe `TunedThresholdClassifierCV` nas versões compatíveis do scikit-learn.
- `prediction.py`: carregamento de artefatos, `predict_proba()` e conversão para `predict()` usando threshold.
- `evaluation.py`: accuracy, precision, recall, F1, AUPRC, ROC-AUC, relatório por classe e matriz de confusão.
- `visualization.py`: figuras da análise exploratória, ROC, precision-recall e matriz de confusão.

O ambiente é excluído do Git. Para criar uma cópia local, use `python -m venv venv` e instale `python -m pip install -e ".[notebooks,test]"`.

### Fluxo dos dados e validação

```text
                       DATASET
                          │
                          ▼
                        data.py
                          │
                          ▼
                  preprocessing.py
                          │
                          ▼
                        X / y
                          │
                  train_test_split
                          │
          ┌────┴───────┐
          ▼            ▼
        TRAIN       TEST (20%)
          │          intocado
          ├── CV / GridSearch / RandomizedSearch
          ├── OOF (diagnóstico do treino)
          ▼
     melhor modelo
          │
       fit(train)
          │
          ▼
    VALIDATION (20%)
          │
    Threshold F2 (β=2)
          │
          ▼
     modelo + threshold
          │
          ▼
    TEST (avaliação final)
          │
    evaluation.py
          │
      metrics.json
```

GridSearchCV e RandomizedSearchCV estão disponíveis como utilitários para busca de hiperparâmetros; a execução atual compara candidatos por CV no treino e escolhe o threshold maximizando F2 na validação. `TunedThresholdClassifierCV` está disponível em `validation.py` como alternativa baseada em validação cruzada.

## Método e métricas

O alvo é codificado como benigno `0` e maligno `1`. O split estratificado usa `random_state=42` e separa 60% para treino, 20% para validação e 20% para teste. A validação escolhe o threshold; o teste fica intocado até a avaliação final.

### O que são as probabilidades OOF

OOF significa *out-of-fold* (fora do fold). Neste projeto, `cross_val_predict` divide somente o conjunto de treino em cinco partes. Para cada parte, o XGBoost é ajustado nas outras quatro e gera probabilidades para a parte que ficou de fora. Ao fim, cada observação de treino tem uma probabilidade produzida por um modelo que não foi ajustado com aquela observação. O parâmetro `method="predict_proba"` retorna as probabilidades das classes; `[:, 1]` seleciona a probabilidade de malignidade (`1`).

Isso dá uma estimativa mais realista do desempenho no treino do que prever sobre as mesmas observações usadas no ajuste, que tende a ser otimista. Neste fluxo, OOF é usado para diagnóstico do treino e não escolhe o threshold. O threshold é escolhido com as probabilidades do conjunto de validação, separado do treino. A regra maximiza **F2** para a classe maligna: o F-beta é a média harmônica ponderada de precision e recall e, com β=2, dá ao recall peso quatro vezes maior que precision ([documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)). O F1 também é reportado como medida equilibrada padrão. O modelo é ajustado no treino, avaliado na validação para escolher o threshold e, depois, avaliado uma única vez no teste reservado.

OOF não equivale a uma validação externa independente: os exemplos ainda pertencem ao conjunto de treino e as estimativas dependem do split e dos hiperparâmetros. O teste reservado fornece uma verificação separada do processo, mas um conjunto de dados externo seria necessário para avaliar generalização fora desta amostra.

Há uma ressalva na leitura das métricas da validação: como o threshold é escolhido usando seus rótulos, as métricas de validação nesse threshold podem ficar otimistas pela própria seleção. O teste reservado não participa da escolha e é a estimativa final mais independente desta divisão. OOF, validação e teste têm funções diferentes; nenhum substitui validação externa.

### Estatísticas da execução local

Resultados calculados com `data/raw/cancer_mama.csv` e split estratificado de 60% treino (341 amostras), 20% validação (114) e 20% teste (114). O XGBoost foi selecionado por acurácia média na CV do treino. O threshold **0.2247** foi escolhido na validação maximizando F2 (β=2), priorizando recall; o mesmo threshold foi aplicado ao OOF e ao teste.

| Métrica | OOF no treino (5 folds) | Validação (escolha do threshold) | Teste reservado |
|---|---:|---:|---:|
| Threshold aplicado | 0.2247 | 0.2247 | 0.2247 |
| AUC-ROC | 0.9874 | 0.9987 | 0.9947 |
| AUPRC | 0.9845 | 0.9978 | 0.9934 |
| Accuracy | 0.9355 | 0.9912 | 0.9737 |
| Precision maligno (1) | 0.8777 | 0.9773 | 0.9535 |
| Recall maligno (1) | 0.9606 | 1.0000 | 0.9762 |
| F1 maligno (1) | 0.9173 | 0.9885 | 0.9647 |
| F2 maligno (1) | 0.9428 | 0.9954 | 0.9716 |
| Matriz de confusão | `[[197, 17], [5, 122]]` | `[[70, 1], [0, 43]]` | `[[70, 2], [1, 41]]` |

OOF descreve previsões fora do fold sobre o conjunto de treino; validação é usada para selecionar o threshold e pode apresentar métricas otimistas por essa seleção. O teste reservado não participa da escolha do modelo nem do threshold e representa a avaliação final desta divisão.

O relatório mantém **precision, recall, F1-score, support, accuracy, macro avg e weighted avg**, além de **AUC-ROC, matrizes de confusão, curva ROC e validação cruzada**. Os resultados podem variar se o dataset ou a configuração do modelo mudar.

## Figuras existentes

As figuras originais ficam diretamente em `images/`:

![Curva ROC](images/curva_roc.png)
![Boxplot](images/boxplot.png)
![Correlação das variáveis](images/correlacao_variaveis.png)
![Distribuição das classes](images/distribuicao_classes.png)
![Distribuição dos dados](images/distribuicao_dados.png)

As figuras de avaliação mais recentes, incluindo a curva Precision-Recall e a matriz de confusão, são geradas em `reports/figures/` ao executar o notebook de treinamento.

## Reprodutibilidade e manutenção

- Configurações do modelo e caminhos ficam centralizados no pacote `src/breast_cancer`.
- Os dados brutos, modelos, logs e resultados gerados são ignorados pelo Git; não coloque dados sensíveis no repositório.
- O XGBoost usa `random_state` explícito e o split é estratificado.
- O threshold não é escolhido no conjunto de teste.
- Para uso real, seria necessária validação externa, análise clínica e governança apropriada.
