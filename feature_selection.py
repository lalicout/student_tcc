# ======================================
# Módulo: feature_selection.py
# ======================================
"""
Seleção de atributos e análise de regressão.

Este módulo fornece:
  - Cálculo de VIF para detecção de multicolinearidade.
  - Seleção de variáveis nominais e ordinais via testes estatísticos.
  - Seleção stepwise de atributos.
  - Ajuste de regressão linear múltipla.
  - Avaliação de resíduos de regressão.

Functions:
    calcular_vif(df: pd.DataFrame, variaveis: List[str]) → pd.DataFrame
    selecionar_nominais_relevantes(df, categoria_de_interesse, variaveis_categoricas, c_c: float = 0.3) → List[str]
    selecionar_ordinais_relevantes(df, variaveis_ordinais, target) → pd.DataFrame
    stepwise_selection(df, target, variaveis_candidatas, threshold_in: float = 0.05, threshold_out: float = 0.1) → List[str]
    regressao_multipla(df, target: str, variaveis: List[str]) → RegressionResults
    ajustar_regressao(df, target_column: str, top_n: int = 10) → Tuple[RegressionResults, List[str], pd.Series, pd.Series]
    selecionar_atributos_results_regressao(resultados, top_n: int = 10) → List[str]
    avaliar_residuos_regressao(y_true, y_pred, nome_modelo: str = 'modelo', materia=None, salvar: bool = False) → None
"""

from eda_functions import aplicar_estilo_visual
from pre_modelagem import *

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, kruskal, shapiro, zscore, entropy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from IPython.display import display

# -------------------------------------------------------------------------------
# MULTICOLINEARIDADE
# -------------------------------------------------------------------------------


def relatorio_multicolinearidade(df, limite_vif=5.0, limite_corr=0.7):
    """
    Gera um relatório de multicolinearidade com base no VIF (Variance Inflation Factor) 
    e na correlação entre variáveis numéricas do DataFrame.

    O relatório combina o valor do VIF com a identificação de pares altamente 
    correlacionados, listando para cada variável com quem ela apresenta 
    correlação forte (acima do limite definido). Útil para diagnósticos de 
    redundância entre preditores antes da modelagem.

    Args:
        df (pd.DataFrame): DataFrame contendo apenas variáveis preditoras 
            numéricas (já codificadas, sem variável target).
        limite_vif (float, optional): Valor mínimo de VIF a partir do qual 
            a variável será considerada com multicolinearidade elevada. 
            Default é 5.0.
        limite_corr (float, optional): Valor mínimo de correlação (em módulo) 
            para identificar variáveis fortemente correlacionadas. 
            Default é 0.7.

    Returns:
        resumo (pd.DataFrame): Tabela com colunas:
            - 'variavel': nome da variável
            - 'vif': valor do VIF
            - 'Alta correlação com': variáveis correlacionadas acima do limite
            - 'avaliacao': alerta textual ('VIF alto', 'Correlação elevada', etc.)
        
        pares_correlacionados (pd.DataFrame): Lista dos pares de variáveis com
            correlação absoluta maior ou igual ao limite definido.
    """
    import numpy as np
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # === 1. Seleciona numéricas e calcula VIF
    X = df.select_dtypes(include=[np.number]).dropna()
    colunas = X.columns
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    df_vif = pd.DataFrame({'variavel': colunas, 'vif': vifs})

    # === 2. Matriz de correlação absoluta
    corr = X.corr().abs()

    # === 3. Pega apenas os pares abaixo da diagonal
    corr_df = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool)).stack().reset_index()
    corr_df.columns = ['variavel_1', 'variavel_2', 'correlacao']

    # === 4. Filtra pares com correlação alta
    pares_correlacionados = corr_df[corr_df['correlacao'] >= limite_corr]
    display(pares_correlacionados)
    # === 5. Para cada variável, lista com quem ela se correlaciona fortemente
    relacoes = (
        pd.concat([
            pares_correlacionados[['variavel_1', 'variavel_2']],
            pares_correlacionados[['variavel_2', 'variavel_1']].rename(
                columns={'variavel_2': 'variavel_1', 'variavel_1': 'variavel_2'}
            )
        ])
        .groupby('variavel_1')['variavel_2']
        .apply(lambda x: ', '.join(sorted(x.unique())))
        .reset_index()
        .rename(columns={'variavel_1': 'variavel', 'variavel_2': 'Alta correlação com'})
    )

    # === 6. Junta VIF + info de pares correlacionados
    resumo = pd.merge(df_vif, relacoes, on='variavel', how='left')
    resumo['Alta correlação com'] = resumo['Alta correlação com'].fillna('—')

    # === 7. Sugestão de alerta
    def avaliar(row):
        if row['vif'] >= limite_vif and row['Alta correlação com'] != '—':
            return 'VIF alto + correlação alta'
        elif row['vif'] >= limite_vif:
            return 'VIF elevado'
        elif row['Alta correlação com'] != '—':
            return 'Correlação elevada'
        else:
            return 'Sem alerta'

    resumo['avaliacao'] = resumo.apply(avaliar, axis=1)

    return resumo.sort_values(by='vif', ascending=False), pares_correlacionados





def calcular_vif(df, variaveis):
    """
    Calcula o VIF (Variance Inflation Factor) para um conjunto de variáveis independentes.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        variaveis (list of str): Lista com os nomes das variáveis independentes.

    Returns:
        pd.DataFrame: DataFrame com os valores de VIF para cada variável.
    """
    X = df[variaveis].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['variavel'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data[vif_data['variavel'] != 'const']


# -------------------------------------------------------------------------------
# SELEÇÃO ESTATÍSTICA
# -------------------------------------------------------------------------------


def selecionar_nominais_relevantes(df,categoria_de_interesse, variaveis_categoricas, c_c=0.3):
    """
    Seleciona variáveis nominais relevantes com base em teste qui-quadrado e coeficiente de contingência.

    Args:
        df (pandas.DataFrame): Conjunto de dados.
        categoria_de_interesse (str): Variável alvo (target).
        variaveis_categoricas (list): Lista de variáveis categóricas nominais.
        c_c (float, optional): Valor mínimo do coeficiente de contingência.

     Returns:        
        list: Lista de nomes das variáveis categóricas relevantes que atendem aos critérios de significância (P-Value < 0.05 e coeficiente de contingência > c_c) 
    
    """
    # Lista para armazenar os resultados temporários
    results = []

    for column in variaveis_categoricas:
        # Tabela de contingência entre a variável categórica e a variável de saída
        contingency_table = pd.crosstab(df[column], df[categoria_de_interesse])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        
        # Calcula o coeficiente de contingência
        n = contingency_table.sum().sum()
        r, c = contingency_table.shape
        contingency_coefficient = (chi2 / (n * min(r - 1, c - 1))) ** 0.5

        # Armazena os resultados para variáveis que passam no teste de P-Value
        if p < 0.05:
            results.append({'Variable': column, 'Chi2': chi2, 'P-Value': p, 'Contingency Coefficient': contingency_coefficient})

    # Cria DataFrame com os resultados e filtrar com base no coeficiente de contingência
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['Contingency Coefficient'] > c_c]
    results_df = results_df.sort_values(by='Contingency Coefficient', ascending=False)
    results_df = results_df.applymap(
                            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
                            else round(x, 3) if isinstance(x, (float, int))
                            else x
                            )

    if not results_df.empty and len(results_df) > 1:

        # Imprime os resultados
        print("Variáveis com P-Value < 0.05 e Coeficiente de Contingência > {:.2f}, ordenadas por Coeficiente de Contingência:".format(c_c))
        display(results_df[['Variable', 'P-Value', 'Contingency Coefficient']])
        
        return results_df['Variable'].tolist()
    
    else: 
        print("Nenhuma dependência significativa foi identificada com base nos critérios estabelecidos.")


# ------------------------------------------------------------------------------------------------


def selecionar_ordinais_relevantes(df, variaveis_ordinais, target):
    """
    Seleciona variáveis ordinais relevantes com base em correlação de Spearman e teste de Kruskal-Wallis.

    Args:
        df (pandas.DataFrame): Dados de entrada.
        variaveis_ordinais (list): Lista de variáveis ordinais.
        target (str): Nome da variável alvo.

    Returns:
        pd.DataFrame: DataFrame contendo as variáveis ordinais relevantes, ordenadas pela força da correlação de Spearman.
                      As colunas incluem:
                      - 'Variável': Nome da variável ordinal.
                      - 'Correlação (Spearman)': Coeficiente de correlação de Spearman.
                      - 'P-valor (Spearman)': P-valor associado ao coeficiente de Spearman.
                      - 'Estatística H (Kruskal)': Estatística H do teste de Kruskal-Wallis.
                      - 'P-valor (Kruskal)': P-valor associado ao teste de Kruskal-Wallis.
    """

    resultados = []

    for var in variaveis_ordinais:
        # Spearman
        coef, p_spear = spearmanr(df[var], df[target])

        # Kruskal-Wallis
        grupos = [df[df[var] == valor][target] for valor in df[var].dropna().unique()]
        h_stat, p_kruskal = kruskal(*grupos)

        resultados.append({
            'Variável': var,
            'Correlação (Spearman)': coef,
            'P-valor (Spearman)': p_spear,
            'Estatística H (Kruskal)': h_stat,
            'P-valor (Kruskal)': p_kruskal
        })

    resultados_df = pd.DataFrame(resultados)

    # Filtrar por significância no Spearman
    resultados_df = resultados_df[resultados_df['P-valor (Spearman)'] < 0.05]

    # Ordenar por força da correlação sem exibir o valor absoluto
    resultados_df = resultados_df.reindex(resultados_df['Correlação (Spearman)'].abs().sort_values(ascending=False).index)

    resultados_df = resultados_df.applymap(
        lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
        else round(x, 3) if isinstance(x, (float, int))
        else x
        )

    # Exibe os resultados ordenados por força da correlação de Spearman
    print("Variáveis ordinais relevantes com base nos testes estatísticos:")
    display(resultados_df)

    return resultados_df


# -------------------------------------------------------------------------------
# SELEÇÃO STEPWISE
# -------------------------------------------------------------------------------


def stepwise_selection(df, target, variaveis_candidatas, threshold_in=0.05, threshold_out=0.1):
    """
    Realiza seleção de variáveis pelo método Stepwise.

    Args:
        df (pandas.DataFrame): DataFrame contendo os dados.
        target (str): Nome da variável dependente (target).
        variaveis_candidatas (list of str): Lista com os nomes das variáveis candidatas à seleção.
        threshold_in (float, optional): Valor p-valor máximo para entrada de variáveis. Padrão é 0.05.
        threshold_out (float, optional): Valor p-valor máximo para remoção de variáveis. Padrão é 0.1.

    Returns:
        list: Lista das variáveis selecionadas pelo processo Stepwise.
    """
    variaveis_selecionadas = []
    
    while True:
        aux_bool = False

        # Testa adicionar variáveis
        variaveis_excluidas = list(set(variaveis_candidatas) - set(variaveis_selecionadas))
        novos_pvalores = pd.Series(index=variaveis_excluidas, dtype=float)
        
        for nova_variavel in variaveis_excluidas:
            modelo_auxiliar = sm.OLS(df[target], sm.add_constant(df[variaveis_selecionadas + [nova_variavel]])).fit()
            novos_pvalores[nova_variavel] = modelo_auxiliar.pvalues[nova_variavel]

        melhor_pvalor = novos_pvalores.min()
        if melhor_pvalor < threshold_in:
            melhor_variavel = novos_pvalores.idxmin()
            variaveis_selecionadas.append(melhor_variavel)
            aux_bool = True

        # Testa remover variáveis
        modelo_atual = sm.OLS(df[target], sm.add_constant(df[variaveis_selecionadas])).fit()
        pvalores_atuais = modelo_atual.pvalues.iloc[1:]  # Ignora a constante
        pior_pvalor = pvalores_atuais.max()
        
        if pior_pvalor > threshold_out:
            pior_variavel = pvalores_atuais.idxmax()
            variaveis_selecionadas.remove(pior_variavel)
            aux_bool = True

        if not aux_bool:
            break

    return variaveis_selecionadas


# -------------------------------------------------------------------------------
# AJUSTE DE REGRESSÃO
# -------------------------------------------------------------------------------


def regressao_multipla(df, target, variaveis):
    """
    Executa uma regressão linear múltipla entre variáveis independentes e a variável alvo.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        target (str): Nome da variável dependente.
        variaveis (list of str): Lista com os nomes das variáveis independentes.

    Returns:
        RegressionResults: Objeto do modelo ajustado de regressão linear.
    """
    X = df[variaveis].dropna()
    y = df[target].loc[X.index]
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    return modelo


def ajustar_regressao(df, target_column, top_n=10):
    """
    Ajusta um modelo de regressão linear aos dados fornecidos e seleciona as variáveis mais relevantes.
    
    Args:

        df (pandas.DataFrame): O DataFrame contendo os dados para análise.
        target_column (str): O nome da coluna que será usada como variável dependente (Y).
        top_n (int, opcional): O número de variáveis independentes mais relevantes a serem selecionadas com base nos p-values. 
                               O padrão é 10.
     Returns:
        tuple: 
            - resultados (statsmodels.regression.linear_model.RegressionResultsWrapper): 
              O objeto contendo os resultados do modelo de regressão ajustado.
            - atributos_relevantes (list): 
              Uma lista com os nomes das variáveis independentes mais relevantes, ordenadas por significância estatística.
            - y_true (pandas.Series): Os valores reais da variável dependente.
            - y_pred (pandas.Series): Os valores previstos pelo modelo.
    """

    # Separa as variáveis dependentes (Y) e independentes (X)
    X = df.drop(columns=[target_column], axis=1)
    Y = df[target_column]

    # Adiciona uma constante para o modelo
    X = sm.add_constant(X)

    # Ajusta o modelo de regressão
    modelo = sm.OLS(Y, X)
    resultados = modelo.fit()

    # Exibe o resumo dos resultados
    print(resultados.summary())

    # Obtem as variáveis mais relevantes usando p-values
    atributos_relevantes = selecionar_atributos_results_regressao(resultados, top_n)

    print("Atributos de maior relevância de acordo com os p-values (ordenados por significância):")
    for i, atributo in enumerate(atributos_relevantes, start=1):
        print(f"{i}. {atributo}")
    # Retorna os valores reais (y_true) e previstos (y_pred) do modelo
    y_true = Y
    y_pred = resultados.predict(X)

    return resultados, atributos_relevantes, y_true, y_pred


#------------------------------------------------------------------------------------------------


def selecionar_atributos_results_regressao(resultados, top_n=10):
    """
    Seleciona os atributos mais relevantes de um modelo de regressão com base nos valores de p-value.
    
    Args:
    
        resultados (statsmodels.regression.linear_model.RegressionResultsWrapper): 
            Objeto contendo os resultados do modelo de regressão ajustado.
        top_n (int, opcional): Número de variáveis a serem selecionadas com base nos menores p-values. 
            O padrão é 10.
    
     Returns:
        list: Lista com os nomes das variáveis selecionadas, ordenadas pelos menores p-values.
    """

    # Extrair o resumo dos resultados como uma tabela
    summary_table = resultados.summary2().tables[1]
    
    # Remover a constante da tabela
    if 'const' in summary_table.index:
        summary_table = summary_table.drop(index='const')
    
    # Ordenar pela coluna de p-value
    sorted_table = summary_table.sort_values('P>|t|', ascending=True)
    
    # Selecionar as top N variáveis
    top_variables = sorted_table.head(top_n)

    top_variables = top_variables.applymap(
        lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
        else round(x, 3) if isinstance(x, (float, int))
        else x
        )
        
    # Imprimir as top N variáveis
    display(top_variables[['Coef.', 'Std.Err.', 't', 'P>|t|']])
    
    # Retornar a lista com os nomes das variáveis
    return top_variables.index.tolist()


# -------------------------------------------------------------------------------
# AVALIAÇÃO DE RESÍDUOS
# -------------------------------------------------------------------------------

def avaliar_residuos_regressao(y_true, y_pred, nome_modelo='modelo', materia=None, salvar=False):
    """
    Avalia os resíduos de um modelo de regressão por meio de visualizações e testes estatísticos.

    Args:
        y_true (array-like): Valores reais da variável dependente.
        y_pred (array-like): Valores preditos pelo modelo.
        nome_modelo (str, optional): Nome do modelo para identificação dos gráficos. Padrão é 'modelo'.
        materia (str, optional): Categoria da análise ('por', 'mat' ou outra). Padrão é None.
        salvar (bool, optional): Indica se os gráficos devem ser salvos. Padrão é False.

    Returns:
        pd.DataFrame: DataFrame contendo resultados dos testes estatísticos dos resíduos.
    """
    if materia == 'portugues':
        materia = ' - Português'
        cor = aplicar_estilo_visual('azul')[3]
        cmap_cor = aplicar_estilo_visual('azul',retornar_cmap=True)
    elif materia == 'matematica':
        materia = ' - Matemática'
        cor = aplicar_estilo_visual('verde')[3]
        cmap_cor = aplicar_estilo_visual('verde',retornar_cmap=True)

    elif materia is None:
        materia = ''
        cor = 'gray'
    else:
        materia = materia
        cor = 'gray'

    residuos = y_true - y_pred
    media_res = np.mean(residuos)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    

    axs[0, 0].scatter(y_pred, residuos, c=y_pred, cmap = cmap_cor, alpha=0.7)
    axs[0, 0].set_title('Resíduos vs Valores Preditos')

    sns.histplot(residuos, kde=True, ax=axs[0, 1], color=cor)
    axs[0, 1].set_title('Distribuição dos Resíduos')

    qq = sm.qqplot(residuos, line='45', ax=axs[0, 2], fit=True)
    qq.axes[0].collections[0].set_color(cor)
    axs[0, 2].set_title('QQ Plot dos Resíduos')


    axs[1, 0].plot(residuos.values, marker='o', linestyle='-',color =cor)
    axs[1, 0].axhline(0, color='red', linestyle='--')
    axs[1, 0].set_title('Resíduos ao dfo da ordem dos dados')

    sns.boxplot(x=residuos, ax=axs[1, 1], color=cor)
    axs[1, 1].set_title('Boxplot dos Resíduos')

    axs[1, 2].scatter(y_true, residuos, alpha=0.7,color = cor)
    axs[1, 2].axhline(0, color='red', linestyle='--')
    axs[1, 2].set_title('Resíduos vs Valor Real')

    plt.suptitle(f'Análise dos Resíduos - {nome_modelo}{materia}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if salvar:
        plt.savefig(f"analise_residuos_{nome_modelo.lower().replace(' ', '_')}{materia}.png", dpi=300)

    plt.show()

    stat_shapiro, p_shapiro = shapiro(residuos)
    X_bp = sm.add_constant(y_pred)
    test_bp = het_breuschpagan(residuos, X_bp)
    dw = durbin_watson(residuos)
    z_scores = np.abs(zscore(residuos))
    outliers = (z_scores > 3).sum()

    resultados = {
        'Média dos Resíduos': [media_res],
        'Shapiro-Wilk Estatística': [stat_shapiro],
        'Normalidade (Shapiro-Wilk)': ['Normal' if p_shapiro > 0.05 else 'Não normal'],
        'p-valor Breusch-Pagan': [test_bp[1]],
        'Homoscedasticidade': ['Homoscedástico' if test_bp[1] > 0.05 else 'Heteroscedástico'],
        'Durbin-Watson': [dw],
        'Autocorrelação': ['OK' if 1.5 < dw < 2.5 else 'Possível autocorrelação'],
        'Outliers (|z| > 3)': [outliers]
    }

    df_resultados = pd.DataFrame(resultados)
    return df_resultados

#------------------------------------------------------------------------------------------------------



def add_features_describe_pd(df,colunas, estudo_frequencia = False,
                              shapiro_values = True, dict_input = None, shannon = False):
    """
    Gera estatísticas descritivas para colunas numéricas e/ou categóricas (ordinais e nominais)
    de um DataFrame, com métricas compatíveis com padrões acadêmicos e ABNT.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados.
        colunas (list): Lista das colunas a serem analisadas.
        estudo_frequencia (bool): Se True, analisa frequencias variáveis categóricas 
            (nominais ou ordinais).
        shapiro_values (bool): Se True, inclui o p-valor do teste de Shapiro-Wilk.
        dict_iput (dict, opcional): Dicionário para Renomear tabelas, 
            caso queira alterar o padrão estabelecido 

    Retorna:
        resumo (pd.DataFrame): resumo das Estatísticas
    """
    if dict_input: 
        nomes_describe_padrao = dict_input
    else:
        nomes_describe_padrao = {
            # Numéricas
            'count': 'Contagem',
            'mean': 'Média',
            'std': 'Desvio Padrão',
            'min': 'Mínimo',
            '25%': '1º Quartil (25%)',
            '50%': 'Mediana (50%)',
            '75%': '3º Quartil (75%)',
            'max': 'Máximo',
            'Shapiro (p)': 'Shapiro-Wilk (p-valor)',
            'CV': 'Coeficiente de Variação (CV)',

            # Categóricas
            'unique': 'Total de Categorias',
            'top': 'Categoria Mais Comum(CMC)',
            'freq': 'Frequência Absoluta CMC',
            'freq rel. top (%)': 'Frequência Relativa CMC (%)',
            '% únicas': 'Diversidade de Categorias (%)'
        }

    
    if estudo_frequencia:
        # Garante tipo string e faz o describe de categóricas
        cat_df = df[colunas].astype(str)
        resumo = cat_df.describe(include=['object']).T

        print(f"Tamanho da amostra: {resumo['count'].unique()[0]}")

        # Frequência relativa da moda e % de categorias únicas
        resumo['freq rel. top (%)'] = (resumo['freq'] / resumo['count'] * 100)
        resumo['% únicas'] = (resumo['unique'] / resumo['count'] * 100)
        
        if entropy:
            resumo['Entropia (Shannon)'] = cat_df.apply(
                lambda x: entropy(x.value_counts(normalize=True), base=2)
                )
        # Renomeia colunas
        resumo.rename(columns=nomes_describe_padrao, inplace=True)
        resumo.drop(columns =['Contagem'],inplace = True)
  
    else:
        # Describe de numéricas
        resumo = df[colunas].describe().T
        print(f"Tamanho da amostra: {resumo['count'].unique()[0]}")
        
        # Moda, Shapiro e CV
        modas = df[colunas].mode().iloc[0]
        resumo['Moda'] = modas
        
        if shapiro_values:
            resumo['Shapiro (p)'] = [f"{shapiro(df[c])[1]:.2e}" for c in colunas]

        resumo['CV'] = resumo['std'] / resumo['mean']

        # Renomeia colunas
        resumo.rename(columns=nomes_describe_padrao, inplace=True)
        resumo.drop(columns =['Contagem'],inplace = True)

    return resumo






def avaliacao_variacao_pontuacao_media_por_categoria(df,atributos, coluna_avaliada='nota_final', marcar_alerta=True):
    """
    Filtra variáveis categóricas relevantes para análise de perfil com base em:
        - Frequência absoluta mínima
        - Frequência relativa mínima (ajustada ao número de categorias)
        - Entropia de Shannon (diversidade da distribuição)
        - Gap de desempenho entre categorias
    E retorna um escore composto de perfilamento.

    Parâmetros:
        df_describe (DataFrame): Tabela do describe com estudo de frequência.
        df (DataFrame): Base original com as notas e variáveis categóricas.
        coluna_avaliada (str): Nome da coluna de nota final.
        marcar_alerta (bool): Se True, adiciona coluna de alerta de dispersão.

    Retorno:
        DataFrame filtrado com colunas adicionais e ordenado por relevância.
    """

    # Dicionário com nomes para exibição no TCC
    colunas_exibidas_tcc = {
        'Total de Categorias':'Total de Categorias',
        'Categoria Mais Comum(CMC)': 'Categoria Dominante',
        'Frequência Relativa CMC (%)': 'Frequência Relativa Dominante(%)',
        'Entropia Relativa': 'Entropia Normalizada',
        'Gap Desempenho': 'Variação de Desempenho por Categoria',
        'PerilScore': 'Score - Perfil'
        }

    col_freq_abs = 'Frequência Absoluta CMC'
    col_freq_rel = 'Frequência Relativa CMC (%)'
    col_n_cat =   'Total de Categorias'
    col_prop_cat = 'Diversidade de Categorias (%)'
    col_entropia = 'Entropia (Shannon)'

    df_describe = add_features_describe_pd(df,colunas=atributos,estudo_frequencia=True,shannon=True) 
    
    # Estima o tamanho da amostra original
    freq_total = df_describe[col_freq_abs] / (df_describe[col_freq_rel] / 100)
    n_total = int(freq_total.median())
    freq_min = max(int(n_total * 0.01), 3)

    # Frequência relativa mínima
    df_describe['freq_rel_min'] = df_describe[col_n_cat].apply(lambda x: 5.0 if x <= 5 else 2.0)

    # Entropia relativa
    df_describe['Entropia Relativa'] = df_describe[col_entropia] / np.log2(df_describe[col_n_cat].replace(0, np.nan).astype(float))
    entropia_adequada = df_describe['Entropia Relativa'] >= 0.5

    # Gap de desempenho por variável
    gaps = {}
    for col in df_describe.index:
        if col in df.columns:
            medias = df.groupby(col)[coluna_avaliada].mean()
            gaps[col] = medias.max() - medias.min()
        else:
            gaps[col] = np.nan
    df_describe['Gap Desempenho'] = pd.Series(gaps)

    # Filtros
    freq_valida = df_describe[col_freq_abs] >= freq_min
    prop_valida = df_describe[col_freq_rel] >= df_describe['freq_rel_min']

    df_filtrado = df_describe[freq_valida & prop_valida & entropia_adequada].copy()

    # Escore composto padronizado
    escore_entropia = df_filtrado['Entropia Relativa'] / df_filtrado['Entropia Relativa'].max()
    escore_gap = df_filtrado['Gap Desempenho'] / df_filtrado['Gap Desempenho'].max()
    df_filtrado['PerfilScore'] = 0.5 * escore_entropia + 0.5 * escore_gap

    # Alerta de dispersão
    if marcar_alerta:
        dispersao = (df_filtrado[col_prop_cat] > 80.0) & (df_filtrado[col_n_cat] > 2)
        df_filtrado['Alerta Dispersão'] = dispersao.map({True: 'Alta dispersão (>80%)', False: ''})
    
    # Filtro final para selecionar variáveis mais relevantes
    
    df_final = df_filtrado[
        (df_filtrado['Entropia Relativa'] >= 0.6) &
        (df_filtrado['Gap Desempenho'] >= 1.0) &
        ~((df_filtrado[col_freq_rel] > 70.0) & (df_filtrado[col_n_cat] <= 3)) &
        (df_filtrado['Alerta Dispersão'] != 'Alta dispersão (>80%)')
    ].copy()
    # Ordenação
    df_final.sort_values(by='PerfilScore', ascending=False, inplace=True)

    # Filtra e renomeia apenas as colunas selecionadas
    df_final = df_final[[col for col in colunas_exibidas_tcc.keys() if col in df_final.columns]]
    df_final.rename(columns=colunas_exibidas_tcc, inplace=True)

    return df_final.round(3)



# ======================================
# Final do módulo
# ======================================
