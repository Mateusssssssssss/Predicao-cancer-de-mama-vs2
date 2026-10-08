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
│   └── imagens_originais/
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
│   │   ├── oof_predictions.csv
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
2. `notebooks/02_treinamento_e_avaliacao.ipynb`: split estratificado, probabilidades OOF para threshold, treino dos modelos e avaliação final.
3. `notebooks/03_cross_val_predict_oof.ipynb`: demonstração isolada de `cross_val_predict` e comparação OOF/in-sample.

Execute a partir da raiz do repositório. O segundo notebook compara candidatos por acurácia média de validação cruzada, gera previsões OOF do vencedor para selecionar o threshold maximizando F2 e salva os dois candidatos em `models/candidates/`, o modelo final em `models/best_model/`, previsões e análise de threshold em `reports/oof/`, métricas em `reports/metrics.json` e gráficos em `reports/figures/`.

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
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
           TRAIN                      TEST
             │                    (intocado até a avaliação)
             ▼
        validation.py
             │
       ┌─────┼───────────┐
       │     │           │
       CV  GridSearch  RandomizedSearch
       │                 │
       └────── OOF ──────┘
               │
       Threshold tuning
       (maximiza F2; β = 2)
               │
         melhor configuração
               │
            models.py
               │
          modelo final
               │
          prediction.py
               │
              TEST
               │
          evaluation.py
               │
          metrics.json
```

GridSearchCV e RandomizedSearchCV estão disponíveis como utilitários para busca de hiperparâmetros; a execução atual seleciona o candidato por CV e escolhe o threshold via OOF maximizando F2. `TunedThresholdClassifierCV` está disponível em `validation.py` como alternativa baseada em validação cruzada.

## Método e métricas

O alvo é codificado como benigno `0` e maligno `1`. O split usa `test_size=0.4` e `random_state=42`, com estratificação para preservar a proporção das classes. O conjunto de teste (40%) fica separado durante a validação cruzada e a escolha do threshold.

### O que são as probabilidades OOF

OOF significa *out-of-fold* (fora do fold). Neste projeto, `cross_val_predict` divide somente o conjunto de treino em cinco partes. Para cada parte, o XGBoost é ajustado nas outras quatro e gera probabilidades para a parte que ficou de fora. Ao fim, cada observação de treino tem uma probabilidade produzida por um modelo que não foi ajustado com aquela observação. O parâmetro `method="predict_proba"` retorna as probabilidades das classes; `[:, 1]` seleciona a probabilidade de malignidade (`1`).

Isso dá uma estimativa mais realista do desempenho no treino do que prever sobre as mesmas observações usadas no ajuste, que tende a ser otimista. As probabilidades OOF também permitem escolher o threshold sem usar o teste. Neste experimento, a regra seleciona o threshold que maximiza **F2** para a classe maligna nas previsões OOF. O F-beta é a média harmônica ponderada de precision e recall; com β=2, o recall recebe peso quatro vezes maior que precision, mantendo as duas métricas no equilíbrio ([documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)). O F1 continua sendo reportado como medida equilibrada padrão, mas não define a escolha do threshold. Só depois o modelo final é ajustado em todo o treino; o threshold selecionado é então avaliado uma única vez no teste reservado.

OOF não equivale a uma validação externa independente: os exemplos ainda pertencem ao conjunto de treino e as estimativas dependem do split e dos hiperparâmetros. O teste reservado fornece uma verificação separada do processo, mas um conjunto de dados externo seria necessário para avaliar generalização fora desta amostra.

Há uma ressalva na leitura das métricas: como o threshold é escolhido usando rótulos e probabilidades OOF, precision, recall, F1 e F2 OOF no threshold escolhido podem ficar otimistas pela própria seleção. A AUC e AUPRC OOF não dependem desse threshold; para estimar o desempenho da regra completa (incluindo a escolha do threshold), a referência mais independente aqui é o teste reservado.

### Estatísticas da execução local

Resultados calculados com `data/raw/cancer_mama.csv`, split estratificado de 341 observações de treino e 228 de teste e XGBoost configurado pelo projeto. O threshold é selecionado para maximizar F2 (β=2) nas probabilidades OOF.

| Métrica | OOF no treino (5 folds) | Teste reservado |
|---|---:|---:|
| Threshold aplicado | 0.4444 (selecionado por F2 nas OOF) | 0.4444 |
| AUC-ROC | 0.9874 | 0.9972 |
| AUPRC | 0.9845 | 0.9961 |
| Accuracy | 0.9619 | 0.9737 |
| Precision maligno (1) | 0.945 | 0.988 |
| Recall maligno (1) | 0.953 | 0.941 |
| F1 maligno (1) | 0.949 | 0.964 |
| F2 maligno (1) | 0.951 | 0.950 |
| Matriz de confusão | `[[207, 7], [6, 121]]` | `[[142, 1], [5, 80]]` |

Como ilustração da diferença entre prever dados vistos e não vistos por cada fold, a AUC in-sample no treino foi **0.9990**, ante **0.9874** OOF.

O relatório mantém **precision, recall, F1-score, support, accuracy, macro avg e weighted avg**, além de **AUC-ROC, matrizes de confusão, curva ROC e validação cruzada**. Os resultados podem variar se o dataset ou a configuração do modelo mudar.

## Figuras existentes

As figuras originais foram preservadas:

![Curva ROC](reports/figures/curva_roc.png)
![Curva Precision-Recall](reports/figures/curva_pr.png)
![Matriz de confusão](reports/figures/matriz_confusao.png)
![Boxplot](reports/figures/boxplot.png)
![Correlação das variáveis](reports/figures/correlacao_variaveis.png)
![Distribuição das classes](reports/figures/distribuicao_classes.png)
![Distribuição dos dados](reports/figures/distribuicao_dados.png)

## Reprodutibilidade e manutenção

- Configurações do modelo e caminhos ficam centralizados no pacote `src/breast_cancer`.
- Os dados brutos, modelos, logs e resultados gerados são ignorados pelo Git; não coloque dados sensíveis no repositório.
- O XGBoost usa `random_state` explícito e o split é estratificado.
- O threshold não é escolhido no conjunto de teste.
- Para uso real, seria necessária validação externa, análise clínica e governança apropriada.
