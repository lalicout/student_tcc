"""
Módulo modelagem.py 

Objetivo Princiapal: Avaliação e Comparação de Modelos de Classificação Binária.

Este módulo contém funções para avaliar o desempenho de classificadores binários
em conjuntos de dados educacionais, com foco em métricas de desempenho, validação
cruzada, matrizes de confusão e curvas ROC/PR. Também permite balancear os dados
de treino e exportar os resultados para tabelas LaTeX.

Funções:

    - avaliar_classificadores_binarios: Avalia classificadores binários utilizando
    métricas de desempenho, validação cruzada, matrizes de confusão e curvas ROC/PR.

    - comparar_modelos_classificacao_binaria: Compara o desempenho de modelos binários
    com base nos resultados de teste e validação cruzada.

Dependências:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - pre_modelagem (funções balancear_dados e exportar_df_para_latex)
    - eda_visualization (função salvar_figura)
    - IPython.display (função display)
    - Os resultados podem ser exportados para tabelas LaTeX para documentação.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, make_scorer, roc_curve, precision_recall_curve, auc
    )
from eda_functions import aplicar_estilo_visual
from pre_modelagem import *
from documentar_resultados import *
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer
)

def avaliar_classificadores_binarios_otimizados(
    X, y, classificadores, param_spaces=None,
    usar_balanceamento=False, materia='portugues'
):
    """Retorna metrics_df, cv_metrics_df e best_params_df."""
    metrics_df = pd.DataFrame()
    cv_metrics_df = pd.DataFrame()
    best_params_df = pd.DataFrame(columns=['Modelo', 'Melhores Parâmetros'])

    cores = {
        'portugues': ['#AED6F1', '#2E86C1'],
        'matematica': ['#D5F5E3', '#28B463']
    }
    cor_0, cor_1 = cores.get(materia, ['gray', 'blue'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    if usar_balanceamento:
        X_train, y_train = balancear_dados(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorers = {
        'AUC ROC': 'roc_auc',
        'Acurácia': 'accuracy',
        'Precisão(0)': make_scorer(precision_score, pos_label=0, zero_division=0),
        'Precisão(1)': make_scorer(precision_score, pos_label=1, zero_division=0),
        'Recall(0)': make_scorer(recall_score, pos_label=0),
        'Recall(1)': make_scorer(recall_score, pos_label=1),
        'F1 Score (Reprovado)': make_scorer(f1_score, pos_label=0),
        'F1 Score (Macro)': 'f1_macro'
    }

    for nome, base in classificadores.items():
        modelo = base.__class__(**base.get_params())
        params = (param_spaces or {}).get(nome, {})

        # Treino sem otimização
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, 'decision_function'):
            y_prob = modelo.decision_function(X_test)
        else:
            y_prob = y_pred

        sem = {
            'Modelo': f"{nome} Sem Otimizacao",
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão(0)': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            'Precisão(1)': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'Recall(0)': recall_score(y_test, y_pred, pos_label=0),
            'Recall(1)': recall_score(y_test, y_pred, pos_label=1),
            'F1 Score (Reprovado)': f1_score(y_test, y_pred, pos_label=0),
            'F1 Score (Macro)': f1_score(y_test, y_pred, average='macro'),
            'AUC ROC': roc_auc_score(y_test, y_prob) if not np.array_equal(y_pred, y_prob) else np.nan
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([sem])], ignore_index=True)

        cvr = {'Modelo': sem['Modelo']}
        for m, s in scorers.items():
            cvr[f'Validação Cruzada ({m})'] = round(
                cross_val_score(modelo, X, y, cv=cv, scoring=s, n_jobs=-1).mean(), 3
            )
        cv_metrics_df = pd.concat([cv_metrics_df, pd.DataFrame([cvr])], ignore_index=True)

        # Treino com otimização
        if params:
            grid = GridSearchCV(base, params, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            best_params_df = pd.concat([
                best_params_df,
                pd.DataFrame([{'Modelo': nome, 'Melhores Parâmetros': grid.best_params_}])
            ], ignore_index=True)

            y_pred_opt = best.predict(X_test)
            if hasattr(best, 'predict_proba'):
                y_prob_opt = best.predict_proba(X_test)[:, 1]
            elif hasattr(best, 'decision_function'):
                y_prob_opt = best.decision_function(X_test)
            else:
                y_prob_opt = y_pred_opt

            com = {
                'Modelo': f"{nome} Com Otimizacao",
                'Acurácia': accuracy_score(y_test, y_pred_opt),
                'Precisão(0)': precision_score(y_test, y_pred_opt, pos_label=0, zero_division=0),
                'Precisão(1)': precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0),
                'Recall(0)': recall_score(y_test, y_pred_opt, pos_label=0),
                'Recall(1)': recall_score(y_test, y_pred_opt, pos_label=1),
                'F1 Score (Reprovado)': f1_score(y_test, y_pred_opt, pos_label=0),
                'F1 Score (Macro)': f1_score(y_test, y_pred_opt, average='macro'),
                'AUC ROC': roc_auc_score(y_test, y_prob_opt) if not np.array_equal(y_pred_opt, y_prob_opt) else np.nan
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([com])], ignore_index=True)

            cvro = {'Modelo': com['Modelo']}
            for m, s in scorers.items():
                cvro[f'Validação Cruzada ({m})'] = round(
                    cross_val_score(best, X, y, cv=cv, scoring=s, n_jobs=-1).mean(), 3
                )
            cv_metrics_df = pd.concat([cv_metrics_df, pd.DataFrame([cvro])], ignore_index=True)

    return metrics_df, cv_metrics_df, best_params_df


def verificar_overfitting(df_teste, df_cv, limite_diferenca=0.10):
    """Retorna DataFrame com Δ% e diagnóstico por modelo."""
    metricas = [
        "Acurácia", "Precisão(1)", "Precisão(0)",
        "Recall(1)",    "Recall(0)",
        "F1 Score (Reprovado)", "F1 Score (Macro)", "AUC ROC"
    ]
    col_cv = {m: f"Validação Cruzada ({m})" for m in metricas}

    df_t = df_teste.set_index("Modelo")
    df_c = df_cv.set_index("Modelo")
    resultados = []

    for m in df_t.index.intersection(df_c.index):
        t, c = df_t.loc[m], df_c.loc[m]
        diffs, res = [], {"Modelo": m}
        for met in metricas:
            vt, vc = t[met], c[col_cv[met]]
            dif = (vc - vt) / vc if vc else 0
            res[f"Δ {met}"] = f"{100 * dif:.1f}%"
            diffs.append(dif)
        media = sum(diffs) / len(diffs) if diffs else 0
        res["Média Δ (%)"] = f"{100 * media:.1f}%"
        res["Diagnóstico"] = "Overfitting" if media > limite_diferenca else "OK"
        resultados.append(res)

    return pd.DataFrame(resultados)




def comparar_resultados_classificacao(
    df_test, df_cv, metrics=None, materia='portugues', salvar=False
):
    """
    Compara desempenho de modelos binários entre Teste e Validação Cruzada,
    detectando automaticamente as colunas de CV no formato
    "Validação Cruzada (<Métrica>)" e aceitando `metrics` como str ou list.

    Args:
        df_test (pd.DataFrame): colunas ['Modelo', <métricas>].
        df_cv (pd.DataFrame): colunas ['Modelo', 'Validação Cruzada (<métrica>)', ...].
        metrics (str or list of str, optional): nomes das métricas em df_test.
            Se None, usa todas as colunas de df_test exceto 'Modelo'.
        materia (str, optional): para títulos e nomes de arquivo. Default 'portugues'.
        salvar (bool, optional): se True, salva figura PNG e tabela LaTeX.
    Returns:
        pd.DataFrame no formato longo com colunas
        ['Modelo','Métrica','Teste','CV','Diferença (%)'].
    """
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from documentar_resultados import exportar_df_para_latex

    # 1. Normaliza `metrics` para lista
    if isinstance(metrics, str):
        metrics = [metrics]
    elif metrics is None:
        metrics = [c for c in df_test.columns if c != 'Modelo']

    # 2. Extrai mapeamento de métricas de CV
    cv_pattern = re.compile(r'^Validação Cruzada \((.+)\)$')
    cv_map = {
        m.group(1): col
        for col in df_cv.columns
        if (m := cv_pattern.match(col))
    }

    # 3. Verifica correspondência
    faltantes = [m for m in metrics if m not in cv_map]
    if faltantes:
        raise ValueError(f"Colunas de CV não encontradas para: {faltantes}")

    # 4. Seleciona e renomeia
    df_t = df_test[['Modelo'] + metrics].copy()
    df_cv_sel = df_cv[['Modelo'] + [cv_map[m] for m in metrics]].copy()
    df_cv_sel.columns = ['Modelo'] + [f"{m} (CV)" for m in metrics]

    # 5. Merge e formato longo
    df = pd.merge(df_t, df_cv_sel, on='Modelo')
    registros = []
    for m in metrics:
        cv_col = f"{m} (CV)"
        for _, row in df.iterrows():
            registros.append({
                'Modelo': row['Modelo'],
                'Métrica': m,
                'Teste': row[m],
                'CV': row[cv_col],
                'Diferença (%)': (row[cv_col] - row[m]) * 100
            })
    df_comp = pd.DataFrame(registros)

    # Quantidade de modelos distintos em df_comp
    n_modelos = df_comp['Modelo'].nunique()

    # Aplica o estilo azul e obtém a paleta com n_modelos cores
    palette = aplicar_estilo_visual('blue_to_green')

    fig, ax = plt.subplots(figsize=(6.4, 3.5))

    # Plota barras com a paleta definida
    sns.barplot(
        data=df_comp,
        x='Métrica',
        y='Teste',
        hue='Modelo',
        palette=palette,
        ci=None,
        ax=ax
    )

    # Plota pontos de CV usando a mesma paleta
    sns.pointplot(
        data=df_comp,
        x='Métrica',
        y='CV',
        hue='Modelo',
        palette=palette,
        markers='D',
        linestyles='--',
        dodge=0.6,
        ci=None,
        ax=ax
    )

    ax.set_title(f'Métricas Teste vs CV – {materia.capitalize()}')
    ax.set_ylabel('Valor')
    ax.tick_params(axis='x')

    # Move a legenda para fora do plot, à direita
    ax.legend(
        title='Modelo',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()
    return df_comp

