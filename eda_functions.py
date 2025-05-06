# ======================================
# Módulo: eda_functions.py
# ======================================
"""
Análise exploratória de desempenho escolar.

Este módulo fornece funções para:
    - Visualizações (boxplots, heatmaps, gráficos de dispersão);
    - Estatísticas descritivas e detecção de outliers;
    - Testes de normalidade e diagnóstico de resíduos;
    - Identificação de padrões entre grupos extremos;
    - Comparação de desempenho entre categorias.

Funções disponíveis:
    quebrar_rotulo(texto: str, max_palavras: int = 1) → str
        Quebra rótulos longos em múltiplas linhas para melhorar a visualização.

    formatar_titulo(texto: str) → str
        Formata rótulos e títulos aplicando capitalização e correções de acentuação.

    plot_distribuicao(df, notas, paleta, materia=None, nome_arquivo='boxplot_notas', mostrar_media=True)
        Gera boxplots para variáveis quantitativas de notas.

    custom_heatmap(matriz_corr, cores, titulo, n_arq, disciplina)
        Cria e salva um mapa de calor baseado em uma matriz de correlação.

    graficos_desempenho_escolar_por_categoria(df, paleta, coluna, nome_arquivo, diretorio, mat=None)
        Gera gráficos de desempenho escolar organizados por categorias.

    plot_notas_faltas(df, cor, dir, mat)
        Gera gráficos de dispersão para atributos quantitativos.

    resumir_outliers(df) → pd.DataFrame
        Calcula e exibe estatísticas de outliers por coluna numérica.

    perfil_categorico_outliers(df_outliers, df_total, variaveis_categoricas) → dict
        Compara a distribuição de categorias nos outliers com a base total.

    identificar_extremos_comparaveis(df, variavel_numerica, variaveis_categoricas, entrada=None, min_diferenca=0.15, q_limite=0.25) → tuple
        Identifica categorias com diferenças significativas entre grupos extremos.

    plot_top_diferencas_extremos(df_diferencas, materia, q1_lim, q3_lim, n_baixo, n_alto, top_n=10, diretorio='graficos_diferencas_perfil', salvar=True)
        Plota as categorias com maior diferença entre grupos de desempenho (baixo vs alto).

    comparar_materias(df, coluna_categorica, colunas_quantitativas, titulo_base, pasta_destino, cores=None, show_plot=True)
        Gera boxplots comparativos para variáveis categóricas e quantitativas.
"""

# ===============================================================================================================
# ==========================================   IMPORTAÇÃO DE BIBLIOTECAS    =====================================
# ===============================================================================================================


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_hex
import statsmodels.api as sm
from IPython.display import display

import pre_modelagem as pmdl
from documentar_resultados import salvar_figura

from IPython.display import display

# -------------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------------------------------


# -------------------------------------
# ESTILO VISUAL PERSONALIZADO


def aplicar_estilo_visual(paleta, retornar_cmap=False, n=None):
    paletas_predefinidas = {
        'azul': {
            'cores': ["#AED6F1", "#5DADE2", "#3498DB", "#2E86C1", "#1B4F72"],
            'auxiliar': ['#d9ecf2']
        },
        'verde': {
            'cores': ["#CDE9D5", "#84C49B", "#4FA06B", "#2D734C", "#1C4030"],
            'auxiliar': ['#EAF6EE']
        },
        'blue_to_green': {
            'cores': ["#1B4F72", "#2E86C1", "#3498DB", "#5DADE2", "#AED6F1",
                      "#F5FAF8", "#D5F5E3", "#A9DFBF", "#52BE80", "#239B56", "#196F3D"],
            'auxiliar': []
        }
    }

    if isinstance(paleta, str) and paleta in paletas_predefinidas:
        base = paletas_predefinidas[paleta]
        cores_base = base['cores']
        cor_auxiliar = base['auxiliar']
    elif isinstance(paleta, list):
        cores_base = paleta
        cor_auxiliar = []

    if retornar_cmap:
        return LinearSegmentedColormap.from_list("custom_cmap", cor_auxiliar + cores_base)

    if n is None:
        return cores_base
    cmap_aux = LinearSegmentedColormap.from_list("paleta_interp", cor_auxiliar + cores_base)
    return [to_hex(cmap_aux(i / (n - 1))) for i in range(n)] if n > 1 else [cores_base[2]]

# DIMENSIONAMENTO E FONTES


def dimensionar_figuresize(n_colunas, n_linhas, largura_base=6, altura_base=6.2,
                           largura_maxima=None, altura_maxima=None, modo='relatorio'):
    if largura_maxima is None:
        largura_maxima = 6.3 if modo == 'relatorio' else 8.0
    if altura_maxima is None:
        altura_maxima = 4.5 if modo == 'relatorio' else 6.0

    largura_natural = largura_base * n_colunas
    altura_natural = altura_base * n_linhas

    escala_final = min(
        largura_maxima / largura_natural if largura_natural > largura_maxima else 1,
        altura_maxima / altura_natural if altura_natural > altura_maxima else 1
    )

    return largura_natural * escala_final, altura_natural * escala_final

def ajustar_fontsize_por_figsize(figwidth, base_width=6):
    escala = figwidth / base_width
    return {
        'axes.titlesize': min(11, max(8, 13 * escala)),
        'axes.labelsize': min(10, max(7, 11 * escala)),
        'xtick.labelsize': min(7, max(6, 9 * escala)),
        'ytick.labelsize': min(7, max(6, 9 * escala)),
        'legend.fontsize': min(7, max(6, 9 * escala)),
        'legend.title_fontsize': min(7, max(6, 9 * escala)),
        'figure.titlesize': min(13, max(9, 15 * escala))
    }

# --------------------------------------
# PADRONIZAÇÃO DE FIGURAS

def padronizar_figura(n_linhas, n_colunas, new_args=None, include_args_subplot=False,
                      escala=0.8, dinamico=False,salvar=False):
    static_sizes = {
        (1, 1): (6.0, 4.5),
        (1, 2): (6.3, 3.6),
        (1, 3): (6.3, 3.6),
        (2, 2): (6.3, 7),
        (2, 3): (6.3, 5)
    }

    # ATENÇÃO: tick.labelsize ≤ 5
    static_fonts = {
        (1, 1): {'axes.titlesize': 9, 'axes.labelsize': 7, 'xtick.labelsize': 4,
                'ytick.labelsize': 4, 'legend.fontsize': 4, 'figure.titlesize': 10,
                'legend.title_fontsize': 4},
        (1, 2): {'axes.titlesize': 9, 'axes.labelsize': 5, 'xtick.labelsize': 4,
                'ytick.labelsize': 4, 'legend.fontsize': 4, 'figure.titlesize': 10,
                'legend.title_fontsize': 7},
        (1, 3): {'axes.titlesize': 9, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 10,
                'legend.title_fontsize': 7},
        (2, 2): {'axes.titlesize': 9, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 10,
                'legend.title_fontsize': 7},
        (2, 3): {'axes.titlesize': 8, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 9,
                'legend.title_fontsize': 7}
    }


    if dinamico:
        figsize = dimensionar_figuresize(n_colunas=n_colunas, n_linhas=n_linhas)
        font_sizes = ajustar_fontsize_por_figsize(figsize[0])
        # Limita os tick labels mesmo em modo dinâmico
        font_sizes['xtick.labelsize'] = min(font_sizes['xtick.labelsize'], 5)
        font_sizes['ytick.labelsize'] = min(font_sizes['ytick.labelsize'], 5)
        fig, axes = plt.subplots(n_linhas, n_colunas, figsize=figsize, dpi=300, **(new_args or {}))
    else:
        key = (n_linhas, n_colunas)
        if key not in static_sizes:
            return padronizar_figura(n_linhas, n_colunas, new_args, include_args_subplot, escala, dinamico=True)

        fig_w, fig_h = [v * escala for v in static_sizes[key]]
        font_sizes = static_fonts[key]
        fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(fig_w, fig_h), dpi=300, **(new_args or {}))

    for key, size in font_sizes.items():
        plt.rcParams[key] = size

    
    return fig, axes, font_sizes


# ----------------------------------------------
# FORMATADORES DE TEXTO


def titulo_para_snake_case(texto):
    """
    Converte um título em snake_case para uso em nomes de arquivos.
    
    Args:
        texto (str): Título original.
    
    Returns:
        str: Versão em snake_case.
    """
    texto = formatar_titulo(texto)
    texto = re.sub(r'[^\w\s]', '', texto)  # remove acentos e pontuação
    texto = texto.lower().strip().replace(" ", "_")
    return texto

def quebrar_rotulo(texto, max_palavras=1):
    """
    Quebra rótulos longos em duas linhas para melhorar a visualização em gráficos.

    Args:
        texto (str): Rótulo original.
        max_palavras (int): Número máximo de palavras por linha.

    Returns:
        str: Rótulo quebrado em duas linhas, se necessário.
    """
        
    palavras = texto.split()
    if len(palavras) > max_palavras:
        meio = len(palavras) // 2
        return ' '.join(palavras[:meio]) + '\n' + ' '.join(palavras[meio:])
    return texto

# -------------------------------------------------------------------------------

def formatar_titulo(texto):
    """
    Formata rótulos e títulos para gráficos, aplicando capitalização e correções de acentuação.

    Args:
        texto (str): Texto a ser formatado.

    Returns:
        str: Texto corrigido e capitalizado.
    """
    if texto.endswith("_por"):
        texto = texto[:-4] + "_portugues"  
    elif texto.endswith("_mat"):
        texto = texto[:-4] + "_matematica"
    else: 
        texto=texto

    texto_formatado = texto.replace("_", " ").strip().title()

    correcoes_titulos = {
        
        "portugues": "Português",
        "matematica": "Matemática",
        "Portugues": "Português",
        "Matematica": "Matemática",
        "Mae": "Mãe",
        "Saude": "Saúde",
        "Educacao": "Educação",
        "Proximo": "Próximo",
        "Reputacao": "Reputação",
        "Responsavel": "Responsável",
        "Area Da Saude": "Área da Saúde",
        "Outra Profissao": "Outra profissão",
        "Servicos": "Serviços",
        "Professor": "Professor(a)",
        "Dona De Casa": "Dona de casa",
        "Dono De Casa": "Dono de casa",
        "Nao": "Não",
        "Sim": "Sim",
        "Romantico": "Romântico",
        "intencao" : "Intenção",
        "reprovacao": "Reprovação",
        "aprovacao": "Aprovação"
    }

    for errado, certo in correcoes_titulos.items():
        texto_formatado = texto_formatado.replace(errado, certo)

    return texto_formatado

# -------------------------------------------------------------------------------
# VISUALIZAÇÕES GERAIS
# -------------------------------------------------------------------------------


# ---------------------------------
#  BOXPLOTS + COUNTPLOTS
def plot_boxplot_countplot(df, x, y, hue, materia=None, paleta='blue_to_green', salvar=False, nome_arquivo='box_count'):
    """
    Gera visualizações combinadas de boxplot e countplot para uma variável categórica.

    Compara a distribuição da variável quantitativa `y` por grupos definidos em `x`,
    segmentando por `hue`. Exibe também a contagem por categoria para avaliação de distribuição amostral.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        x (str): Coluna categórica para o eixo x.
        y (str): Coluna quantitativa para o eixo y (boxplot).
        hue (str): Coluna categórica para segmentação (usualmente 'aprovacao').
        materia (str): Nome da disciplina, usado no título e no nome do arquivo.
        paleta (str, optional): Nome da paleta de cores. Default é 'verde'.
        salvar (bool, optional): Se True, salva a figura com nome automático. Default é False.
        nome_arquivo (str, optional): Nome base do arquivo salvo. Default é 'box_count'.

    Returns:
        tuple: Matplotlib figure e lista de axes.
    """

    cores = aplicar_estilo_visual(paleta, n=2)
    fig, axes, font_sizes = padronizar_figura(1, 2)
    plt.rcParams.update(font_sizes)


    sns.boxplot(data=df, x=x, y=y, ax=axes[0], palette=cores,
                linewidth=0.8,
                boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                whiskerprops={'color': 'black', 'linewidth': 0.8},
                capprops={'color': 'black', 'linewidth': 0.8},
                medianprops={'color': 'black', 'linewidth': 0.8})
    axes[0].set_title(f"por {formatar_titulo(x)}")
    axes[0].set_xlabel(quebrar_rotulo(formatar_titulo(x)))
    axes[0].set_ylabel(formatar_titulo(y))

    sns.countplot(data=df, x=x, hue=hue, ax=axes[1], palette=cores)
    axes[1].set_title(f"por {formatar_titulo(x)}")
    axes[1].set_xlabel(quebrar_rotulo(formatar_titulo(x)))
    axes[1].set_ylabel("Contagem")
    if hue:
        axes[1].legend(title=formatar_titulo(hue), loc='upper right')

    xticks_texts_0 = [tick.get_text() for tick in axes[0].get_xticklabels()]
    xticks_texts_formatados_0 = [quebrar_rotulo(texto) for texto in xticks_texts_0]
    axes[1].set_xticklabels(xticks_texts_formatados_0, fontsize=4)
    axes[1].tick_params(axis='y', labelsize=4)
    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(ymin, ymax + 50)


    titulo = f"Distribuição de {formatar_titulo(y)} e frequência \n por {formatar_titulo(x)}"
    if materia:
        plt.figtext(
            0.5, -0.02,
            f"Disciplina: {formatar_titulo(materia)}",
            ha='center', fontsize=8, style='italic'
        )
    fig.suptitle(titulo)
    plt.tight_layout()

    if salvar:
        salvar_figura(nome_arquivo, materia if materia else '')

    plt.show()

# --------------------------------------
# BOXPLOTS + BOXPLOTS (duplo)

def plot_boxplot_boxplot(df, x, y1, y2, hue, materia, paleta, salvar=False, nome_arquivo='box_box'):
    cores = aplicar_estilo_visual(paleta, n=2)
    fig, axes, font_sizes = padronizar_figura(1, 2)

    for ax, y in zip(axes, [y1, y2]):
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=cores,
                    linewidth=0.8,
                    boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                    whiskerprops={'color': 'black', 'linewidth': 0.8},
                    capprops={'color': 'black', 'linewidth': 0.8},
                    medianprops={'color': 'black', 'linewidth': 0.8})

        ax.tick_params(axis='x', labelsize=font_sizes['xtick.labelsize'])
        ax.tick_params(axis='y', labelsize=font_sizes['ytick.labelsize'])
        ax.set_title(f"por {formatar_titulo(x)}")
        ax.set_xlabel(quebrar_rotulo(formatar_titulo(x)))
        ax.set_ylabel(formatar_titulo(y))
        if hue:
            ax.legend_.remove()

    notas = {'nota1', 'nota2', 'nota_final',
             'nota1_por', 'nota2_por', 'nota_final_por',
             'nota1_mat', 'nota2_mat', 'nota_final_mat'}
    if y1 in notas and y2 in notas:
        titulo = f"Boxplots das Notas por {formatar_titulo(x)}"
        if materia:
            titulo += f" - {formatar_titulo(materia)}"
    else:
        titulo = f"Distribuição de {formatar_titulo(y1)} e {formatar_titulo(y2)} por {formatar_titulo(x)}"
        if materia:
            plt.figtext(
    0.5, -0.02,
    f"Disciplina: {formatar_titulo(materia)}",
    ha='center', fontsize=8, style='italic'
)

    fig.suptitle(titulo)
    fig.tight_layout()

    if salvar:
        salvar_figura(nome_arquivo, materia if materia else '')

    return fig, axes

# --------------------------------------
# QUANTITATIVAS

# DISTRIBUIÇÃO

def plot_distribuicao_quantitativas(df, colunas, modo='box', mostrar_media=False, mostrar_mediana=False,
                                     titulo=None, paleta='azul', materia=None):
    """
    Visualiza variáveis quantitativas com múltiplos boxplots ou histogramas (histplot) lado a lado.

    Args:
        df (DataFrame): Base de dados.
        colunas (list): Lista de colunas numéricas para análise.
        modo (str): 'box' ou 'hist'. Define o tipo de visual.
        mostrar_media (bool): Se True, anota a média nos boxplots.
        mostrar_mediana (bool): Se True, desenha linha da mediana nos boxplots.
        titulo (str): Título do gráfico. Se None, define automaticamente.
        paleta (str): Paleta usada para as cores dos plots.
        materia (str): Nome da disciplina (para título, opcional).

    Returns:
        tuple: (fig, axes) — figura matplotlib e lista de axes
    """
    
    
    
    if paleta == 'blue_to_green':
        cor = 0
        n = len(colunas)
    else:
        cor= 2
        n = len(colunas)+2
    cores = aplicar_estilo_visual(paleta, n=n)
    fig, axes, _ = padronizar_figura(1, n-cor)

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i, col in enumerate(colunas):
        ax = axes[i]

        if modo == 'box':
            sns.boxplot(data=df, y=col, ax=ax, color=cores[i+cor],
                        linewidth=0.8,
                        boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                        whiskerprops={'color': 'black', 'linewidth': 0.8},
                        capprops={'color': 'black', 'linewidth': 0.8},
                        medianprops={'color': 'black', 'linewidth': 0.8})

            if mostrar_media:
                media = df[col].mean()
                ax.annotate(f"{media:.2f}", xy=(0, media),
                            xytext=(0.1, media),
                            textcoords='data',
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=0.5),
                            fontsize=7)

            if mostrar_mediana:
                mediana = df[col].median()
                ax.axhline(mediana, linestyle='--', color='gray', linewidth=0.8)

            ax.set_xlabel("")
            ax.set_ylabel(formatar_titulo(col))

        elif modo == 'hist':
            ymax = df.shape[0] / 3.33
            sns.histplot(data=df, x=col, kde=True, ax=ax, color=cores[i+cor], bins=20)
            ax.set_xlabel(formatar_titulo(col))
            ax.set_ylabel("Frequência")
            ax.set_ylim(0, ymax)

    # TÍTULO AUTOMÁTICO
    if titulo is None:
        if all('nota' in c for c in colunas):
            titulo = "Distribuição das Notas"
        elif all('falta' in c for c in colunas):
            titulo = "Distribuição das Faltas"
        elif 'idade' in colunas:
            titulo = "Distribuição de Idade e Faltas"
        else:
            titulo = "Distribuições Quantitativas"

    if materia:
        titulo += f" - {formatar_titulo(materia)}"

    fig.suptitle(titulo)
    fig.tight_layout()
    return fig, axes

# MAPAS DE CALOR
def custom_heatmap(matriz_corr, cores, titulo, n_arq=None, disciplina=None,salvar = False):
    """
    Gera e salva um mapa de calor baseado em matriz de correlação.

    Args:
        matriz_corr (pd.DataFrame): Matriz de correlação.
        cores (str | list): Nome da paleta ou lista de cores.
        titulo (str): Título principal do gráfico.
        n_arq (str, optional): Nome base do arquivo. Se None, será gerado a partir do título.
        disciplina (str, optional): Nome da disciplina (ex: 'portugues', 'matematica').

    Returns:
        None
    """
    # Paleta e figura
    cmap_custom = aplicar_estilo_visual(cores, retornar_cmap=True)
    fig, ax, font_sizes = padronizar_figura(1, 1)

    # Plot
    sns.heatmap(matriz_corr,
                annot=True,
                cmap=cmap_custom,
                fmt=".2f",
                annot_kws={"size": 6},
                cbar=False,
                ax=ax)

    # Título visual
    titulo_formatado = formatar_titulo(titulo)
    subtitulo = f"{titulo_formatado}" + (f" - {formatar_titulo(disciplina)}" if disciplina else "")
    ax.set_title(quebrar_rotulo(subtitulo, max_palavras=3),
                 fontsize=11,
                 pad=8)

    # Eixos
    ax.tick_params(axis='both', labelsize=4)

    # Nome do arquivo
    nome_base = titulo_para_snake_case(titulo) if n_arq is None else n_arq

    # Salvamento
    plt.tight_layout()
    if salvar:
        salvar_figura(f"mapa_calor_{nome_base}",
                    diretorio=os.path.join('correlacoes', disciplina) if disciplina else 'correlacoes',
                    materia=disciplina)
    plt.show()




def selecao_impacto_variaveis_categoricas(df, variaveis_categoricas,
                                          paleta = 'azul',
                                          salvar=True,
                                          materia=None,
                                          coluna_avaliada='nota_final'):
    """
    Avalia o impacto de variáveis categóricas com base no gap de desempenho e no desequilíbrio. 
    E plota apenas as variáveis com maior e menor impacto.

    Parâmetros:
        df (DataFrame): Base de dados da matéria.
        variaveis_categoricas (list): Lista de colunas categóricas a avaliar.
        materia (str): Nome da disciplina (ex: 'Matematica' ou 'Portugues').
        diretorio (str): Pasta onde salvar os gráficos.
        coluna_avaliada (str): Nome da coluna com a nota final.
        paleta (str): Paleta de cor usada nos gráficos.
    """

    if materia is None:
        gap_min = 1.0
        frequencia_dominante_max = 70.0

        for col in variaveis_categoricas:

            if df[col].nunique() <= 1:
                continue

            # Média por matéria dentro de cada categoria
            medias_por = df.groupby(col)[f'{coluna_avaliada}_por'].mean()
            medias_mat = df.groupby(col)[f'{coluna_avaliada}_mat'].mean()

            gap_comportamento = abs(medias_por - medias_mat)

            # Frequência dominante
            freq_dominante = df[col].value_counts(normalize=True).max() * 100

            if (
                gap_comportamento.max() >= gap_min and
                freq_dominante <= frequencia_dominante_max
            ):
                nome_do_arquivo = f'{col}_desempenho_fraco.png'
            # gerar gráfico
                plot_boxplot_boxplot(df,
                                       x = col,
                                       materia=materia,
                                       y1=f'{coluna_avaliada}_por',
                                       y2=f'{coluna_avaliada}_mat',
                                       paleta=paleta,
                                       hue=None,
                                       nome_arquivo=nome_do_arquivo,
                                       salvar=salvar)
    else:
        dp = df[coluna_avaliada].std()
        # Define limiares proporcionais ao desvio padrão
        limite_gap_fraco = 0.3 * dp
        limite_gap_forte = 0.9 * dp
        for col in variaveis_categoricas:
            
            n_cat = df[col].nunique()
            if n_cat == 2:
                limiar_desequilibrio = 0.75
            elif n_cat <= 4:
                limiar_desequilibrio = 0.60
            else:
                limiar_desequilibrio = 0.50

            desequilibrio = df[col].value_counts(normalize=True).max()
            medias = df.groupby(col)[coluna_avaliada].mean()
            gap_media = medias.max() - medias.min()

            # Critério de impacto fraco
            if desequilibrio <= limiar_desequilibrio and gap_media <= limite_gap_fraco:
                print(f"[FRACO] {col} → equilíbrio: {desequilibrio:.2f} | gap: {gap_media:.2f}")
                nome_do_arquivo = f'{col}_desempenho_fraco.png'
                plot_boxplot_countplot(df,
                                       x = col,
                                       materia=materia,
                                       paleta=paleta,
                                       y=coluna_avaliada,
                                       hue='aprovacao',
                                       nome_arquivo=nome_do_arquivo,
                                       salvar=salvar)

            # Critério de impacto forte
            elif desequilibrio >= limiar_desequilibrio and gap_media >= limite_gap_forte:
                print(f"[FORTE] {col} → desequilíbrio: {desequilibrio:.2f} | gap: {gap_media:.2f}")
                nome_do_arquivo = f'{col}_desempenho_forte.png'
                plot_boxplot_countplot(df,
                                       x = col,
                                       materia=materia,
                                       y=coluna_avaliada,
                                       paleta=paleta,
                                       hue='aprovacao',
                                       nome_arquivo= nome_do_arquivo,
                                       salvar = salvar)


# -------------------------------------------------------------------------------

def comparar_notas_faltas(df, cor, dir):
    """
    Gera e salva 6 gráficos comparando Português e Matemática:
    Linha 1: Comparação das notas 1, 2 e final entre POR e MAT
    Linha 2: Comparação de faltas + relação entre faltas e nota final
    """
    paleta = aplicar_estilo_visual(cor)
    fig, axes, font_sizes = padronizar_figura(2, 3)

    comparacoes = [
        ('nota1_por', 'nota1_mat', 'POR vs MAT\n Nota 1 '),
        ('nota2_por', 'nota2_mat', 'POR vs MAT \n Nota 2 '),
        ('nota_final_por', 'nota_final_mat', 'POR vs MAT \n Nota Final '),
        ('faltas_por', 'faltas_mat', 'POR vs MAT \n Faltas'),
        ('faltas_por', 'nota_final_por', 'Nota Final vs Faltas \n Português'),
        ('faltas_mat', 'nota_final_mat', 'Nota Final vs Faltas \n Matemática'),
    ]

    for i, (x, y, titulo) in enumerate(comparacoes):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        sns.regplot(data=df, x=x, y=y, ax=ax, color=paleta[2],
                    scatter_kws={'alpha': 0.5, 's': 12})
        ax.set_title(titulo, fontsize=font_sizes['axes.labelsize'])
        ax.set_xlabel(formatar_titulo(x), fontsize=font_sizes['axes.labelsize'])
        ax.set_ylabel(formatar_titulo(y), fontsize=font_sizes['axes.labelsize'])
        ax.tick_params(axis='both', labelsize=font_sizes['xtick.labelsize'])
        ax.grid(False)

    fig.suptitle("Comparação entre Português e Matemática", fontsize=font_sizes['axes.titlesize'])
    plt.tight_layout()
    salvar_figura(diretorio=dir, nome_arquivo="comparativo_por_mat", materia=None)
    plt.show()


def plot_notas_faltas(df, cor, dir, mat):
    """
   Gera e salva gráficos de dispersão para atributos quantitativos.

    Subplots:
        1. Faltas vs Nota Final
        2. Nota 1 vs Nota 2
        3. Nota 1 vs Nota Final
        4. Nota 2 vs Nota Final

    Args:
        df (pd.DataFrame):
            DataFrame com colunas obrigatórias:
            - 'faltas'
            - 'nota1'
            - 'nota2'
            - 'nota_final'
            - 'aprovacao'
        cor (str):
            Nome da paleta de cores a ser aplicada (ex: 'azul', 'verde').
        dir (str):
            Caminho da pasta onde a figura será salva.
        mat (str):
            Código da matéria para título e nome de arquivo
            ('por' para Português, 'mat' para Matemática ou outro).

    Returns:
        None
    """
    # Configurando o padrão visual
    
    paleta = aplicar_estilo_visual(cor)
    
    # Cria a figura
    fig, axes, font_sizes = padronizar_figura(2,2)
    
    # Define a matéria para título e nome do arquivo
    if mat == 'portugues':
        materia = 'Português'
    elif mat == 'matematica':
        materia = 'Matemática'
    elif mat is None:
        materia = ''
    else:
        materia = mat
    
    fig.suptitle(f'Visualização de Atributos Quantitativos - {materia}', fontsize=font_sizes['axes.titlesize'])

    # Gráfico 1: Faltas vs Nota Final
    sns.regplot(data=df, x='faltas', y='nota_final', ax=axes[0, 0], color=paleta[3], scatter_kws={'alpha': 0.5, 's': 12})
    axes[0, 0].set_title('Faltas vs Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[0, 0].set_xlabel('Faltas', fontsize=font_sizes['axes.labelsize'])
    axes[0, 0].set_ylabel('Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[0, 0].tick_params(axis='both', labelsize=font_sizes['xtick.labelsize'])
    axes[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axes[0, 0].grid(False)

    # Gráfico 2: Nota 1 vs Nota 2
    sns.regplot(data=df, x='nota1', y='nota2', ax=axes[0, 1], color=paleta[3], scatter_kws={'alpha': 0.5, 's': 12})
    axes[0, 1].set_title('Nota 1 vs Nota 2', fontsize=font_sizes['axes.labelsize'])
    axes[0, 1].set_xlabel('Nota 1', fontsize=font_sizes['axes.labelsize'])
    axes[0, 1].set_ylabel('Nota 2', fontsize=font_sizes['axes.labelsize'])
    axes[0, 1].tick_params(axis='both', labelsize=font_sizes['xtick.labelsize'])
    axes[0, 1].grid(False)

    # Gráfico 3: Nota 1 vs Nota Final
    sns.regplot(data=df, x='nota1', y='nota_final', ax=axes[1, 0], color=paleta[3], scatter_kws={'alpha': 0.5, 's': 12})
    axes[1, 0].set_title('Nota 1 vs Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[1, 0].set_xlabel('Nota 1', fontsize=font_sizes['axes.labelsize'])
    axes[1, 0].set_ylabel('Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[1, 0].tick_params(axis='both', labelsize=font_sizes['xtick.labelsize'])
    axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    axes[1, 0].grid(False)

    # Gráfico 4: Nota 2 vs Nota Final
    sns.regplot(data=df, x='nota2', y='nota_final', ax=axes[1, 1], color=paleta[3], scatter_kws={'alpha': 0.5, 's': 12})
    axes[1, 1].set_title('Nota 2 vs Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[1, 1].set_xlabel('Nota 2', fontsize=font_sizes['axes.labelsize'])
    axes[1, 1].set_ylabel('Nota Final', fontsize=font_sizes['axes.labelsize'])
    axes[1, 1].tick_params(axis='both', labelsize=font_sizes['xtick.labelsize'])
    axes[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axes[1, 1].grid(False)

    plt.tight_layout()

    # Salva figura após ajustes de layout
    salvar_figura(diretorio=dir, nome_arquivo="plot_notas_faltas", materia=mat)
    
    plt.show()

# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# SUMARIO ESTATÍSTICO
# -------------------------------------------------------------------------------

def resumir_outliers(df):
    """
    Calcula e exibe um DataFrame com estatísticas de outliers por coluna numérica.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: Tabela com Q1, Q3, limites e contagens de outliers,
                      ordenada por maior número de outliers.
    """

    resumo = {}
    colunas = df.select_dtypes(include='number').columns

    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inferior = Q1 - 1.5 * IQR
        lim_superior = Q3 + 1.5 * IQR

        outliers_baixo = df[df[col] < lim_inferior]
        outliers_cima = df[df[col] > lim_superior]

        resumo[col] = {
            'Q1 (1º Quartil)': Q1,
            'Q3 (3º Quartil)': Q3,
            'Limite Inferior (L1)': lim_inferior,
            'Limite Superior (L3)': lim_superior,
            'Outliers Totais': len(outliers_baixo) + len(outliers_cima),
            'Outliers < L1': len(outliers_baixo),
            'Outliers > L3': len(outliers_cima)
        }

    resumo_df = pd.DataFrame.from_dict(resumo, orient='index')
    resumo_df = resumo_df.sort_values(by='Outliers Totais', ascending=False)
    resumo_df = resumo_df.applymap(
        lambda x: (
            0 if isinstance(x, (float, int)) and abs(x) < 1e-10
            else f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
            else round(x, 3) if isinstance(x, (float, int))
            else x
        )
)
    display(resumo_df)

    return resumo_df

def perfil_categorico_outliers(df_outliers, df_total, variaveis_categoricas):
    """
    Compara a distribuição de categorias nos outliers com a base total.

    Gera uma tabela com:
      - frequências absolutas nos outliers;
      - percentuais nos outliers e na base completa;
      - diferença percentual entre os dois grupos.

    Args:
        df_outliers (pd.DataFrame):
            Subconjunto de alunos classificados como outliers.
        df_total (pd.DataFrame):
            Base completa dos dados da disciplina analisada.
        variaveis_categoricas (list of str):
            Lista de colunas categóricas (ordinais ou nominais).

    Returns:
        dict of pd.DataFrame:
            Um dicionário onde cada chave é uma variável categórica
            e o valor é um DataFrame com os percentuais e diferenças relativas.
    """
    perfis = {}

    for var in variaveis_categoricas:
        dist_total = df_total[var].value_counts(normalize=True)
        dist_out = df_outliers[var].value_counts(normalize=True)
        contagem_out = df_outliers[var].value_counts()

        df_result = pd.DataFrame({
            'Frequência Outlier': contagem_out,
            '% Outlier': dist_out,
            '% Total': dist_total
        }).fillna(0)

        df_result['Diferença (%)'] = (df_result['% Outlier'] - df_result['% Total']) * 100
        df_result['% Outlier'] = (df_result['% Outlier'] * 100).round(1).astype(str) + '%'
        df_result['% Total'] = (df_result['% Total'] * 100).round(1).astype(str) + '%'
        df_result['Diferença (%)'] = df_result['Diferença (%)'].round(1)

        perfis[var] = df_result.sort_values(by='Diferença (%)', ascending=False)

    return perfis


def comparar_grupos_extremos(df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca=0.15):
    """
    Compara a distribuição de categorias entre dois grupos extremos com base em uma variável numérica.

    Args:
        df (pd.DataFrame): Base de dados original.
        variavel_numerica (str): Nome da variável usada para segmentar os grupos.
        variaveis_categoricas (list of str): Lista de variáveis categóricas a serem avaliadas.
        Q1 (float): Limite inferior para o grupo de baixo desempenho (nota ≤ Q1).
        Q3 (float): Limite superior para o grupo de alto desempenho (nota ≥ Q3).
        min_diferenca (float, optional): Diferença mínima de proporção entre os grupos para considerar uma categoria relevante. Default é 0.15.

    Returns:
        tuple:
            - pd.DataFrame: Categorias com diferenças relevantes entre os grupos, ordenadas pela diferença.
            - int: Tamanho do grupo de baixo desempenho.
            - int: Tamanho do grupo de alto desempenho.
    """

    grupo_baixo = df[df[variavel_numerica] <= Q1]
    grupo_alto = df[df[variavel_numerica] >= Q3]
    n_baixo, n_alto = len(grupo_baixo), len(grupo_alto)
    resultados = []

    for var in variaveis_categoricas:
        dist_baixo = grupo_baixo[var].value_counts(normalize=True)
        dist_alto = grupo_alto[var].value_counts(normalize=True)
        abs_baixo = grupo_baixo[var].value_counts()
        abs_alto = grupo_alto[var].value_counts()

        for categoria in set(dist_baixo.index).union(dist_alto.index):
            perc_baixo = dist_baixo.get(categoria, 0)
            perc_alto = dist_alto.get(categoria, 0)
            dif = abs(perc_baixo - perc_alto)

            if dif >= min_diferenca:
                resultados.append({
                    'Variável': var,
                    'Categoria': categoria,
                    f'% Grupo Nota Baixa (≤{Q1:.1f})': f"{round(perc_baixo * 100, 1)}% ({abs_baixo.get(categoria, 0)}/{n_baixo})",
                    f'% Grupo Nota Alta (≥{Q3:.1f})': f"{round(perc_alto * 100, 1)}% ({abs_alto.get(categoria, 0)}/{n_alto})",
                    'Diferença Absoluta (%)': round(dif * 100, 1)
                })

    df_resultado = pd.DataFrame(resultados).sort_values(by='Diferença Absoluta (%)', ascending=False)
    df_resultado['Diferença Absoluta (%)'] = df_resultado['Diferença Absoluta (%)'].apply(
        lambda x: f"{x:.2e}" if (abs(x) < 1e-4 or abs(x) > 1e4) else round(x, 3)
    )

    return df_resultado, n_baixo, n_alto


def identificar_extremos_comparaveis(
    df,
    variavel_numerica,
    variaveis_categoricas,
    min_diferenca=None,
    q_limite=None,
    entrada=None,
    otimizar=True
):
    """
    Identifica categorias com diferenças significativas entre grupos extremos de uma variável numérica.

    Args:
        df (pd.DataFrame): Base de dados.
        variavel_numerica (str): Nome da variável contínua.
        variaveis_categoricas (list): Lista de variáveis categóricas a serem comparadas.
        min_diferenca (float, optional): Diferença mínima entre os grupos (%). Default: 0.15.
        q_limite (float, optional): Quartil base para Q1 e Q3. Ignorado se otimizar=True.
        entrada (tuple, optional): Limites manuais (Q1, Q3). Sobrescreve o modo automático.
        otimizar (bool, optional): Se True, busca os melhores limites respeitando critérios. Default: True.

    Returns:
        tuple: (df_diferencas, n_baixo, n_alto, Q1, Q3)
    """

    #Defaults internos 
    if min_diferenca is None:
        min_diferenca = 0.15
    if q_limite is None:
        q_limite = 0.25

    # Modo com entrada manual explícita
    if entrada:
        Q1, Q3 = entrada
        df_dif, n_baixo, n_alto = comparar_grupos_extremos(df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca)
        display(df_dif)
        return df_dif, n_baixo, n_alto, Q1, Q3

    # Modo automático com otimização
    if otimizar:
        melhores_resultados = None
        melhor_diff = float('inf')

        nota_min_sup = 14.0
        nota_max_inf = 10.0
        n_minimo = 30
        max_diff_ratio = 0.2
        q_vals = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]

        for q in q_vals:
            Q1 = df[variavel_numerica].quantile(q)
            Q3 = df[variavel_numerica].quantile(1 - q)

            if Q3 < nota_min_sup or Q1 > nota_max_inf:
                continue

            df_dif, n_baixo, n_alto = comparar_grupos_extremos(df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca)

            if n_baixo < n_minimo or n_alto < n_minimo:
                continue

            if abs(n_baixo - n_alto) / max(n_baixo, n_alto) > max_diff_ratio:
                continue

            diff = abs(n_baixo - n_alto)
            if diff < melhor_diff:
                melhor_diff = diff
                melhores_resultados = (df_dif, n_baixo, n_alto, Q1, Q3)

        if melhores_resultados:
            display(melhores_resultados[0])
            return melhores_resultados
        else:
            print("Nenhuma configuração satisfaz os critérios de otimização.")
            return None, 0, 0, None, None

    # Modo sem otimização
    Q1 = df[variavel_numerica].quantile(q_limite)
    Q3 = df[variavel_numerica].quantile(1 - q_limite)
    df_dif, n_baixo, n_alto = comparar_grupos_extremos(df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca)
    display(df_dif)
    return df_dif, n_baixo, n_alto, Q1, Q3



def plot_top_diferencas_extremos(df_diferencas, materia, q1_lim, q3_lim, n_baixo, n_alto,
                                  top_n=10, diretorio='graficos_diferencas_perfil', salvar=True):
    """
    Plota as categorias com maior diferença entre grupos de desempenho (baixo vs alto).

    Args:
        df_diferencas (pd.DataFrame): Saída da função identificar_extremos_comparaveis.
        materia (str): 'portugues' ou 'matematica'.
        q1_lim (float): Limite inferior usado para notas baixas.
        q3_lim (float): Limite superior usado para notas altas.
        n_baixo (int): Tamanho do grupo de nota baixa.
        n_alto (int): Tamanho do grupo de nota alta.
        top_n (int, optional): Número de categorias para exibir. Default é 10.
        diretorio (str, optional): Pasta onde salvar a imagem. Default 'graficos_diferencas_perfil'.
        salvar (bool, optional): Se True, salva o gráfico com salvar_figura.

    Returns:
        None
    """

    # Seleção e ordenação
    df_top = df_diferencas.copy().head(top_n).copy()
    df_top['rótulo'] = df_top['Variável'] + ' = ' + df_top['Categoria'].astype(str)
    df_top = df_top.sort_values(by='Diferença Absoluta (%)', ascending=False)

    # Estilo visual
    paleta = 'azul' if materia == 'portugues' else 'verde'
    cores = aplicar_estilo_visual(paleta,n=top_n)
    print(cores)
    # Figura
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = sns.barplot(
        y=df_top['rótulo'], 
        x=df_top['Diferença Absoluta (%)'], 
        palette=cores[::-1],
        ax=ax
    )

    # Contraste ajustado nas anotações
    for i, (v, patch) in enumerate(zip(df_top['Diferença Absoluta (%)'], bars.patches)):
        brightness = mcolors.rgb_to_hsv(mcolors.to_rgb(patch.get_facecolor()))[2]
        fator_contraste = 0.45 if materia == 'matematica' or '' else 0.255
        text_color = mcolors.to_hex((brightness * fator_contraste,) * 3) if brightness > 0.72 else 'white'
        ax.text(v / 2, i, f"{v:.1f}%", color=text_color, va='center', ha='center', fontsize=12)

    # Título e subtítulo
    if materia:
        materia_title = f'- {materia}'
    else: 
        materia_title = ''
    
    plt.title(
        f'As {top_n} Categorias com Maior Influência no Desempenho Escolar' + materia_title,
        fontsize=14, weight='bold', pad=15
    )
    plt.figtext(
        0.5, -0.03,
        f'\nCritério: Diferença Absoluta (%) entre Grupos de Baixo (≤{q1_lim:.1f}) e Alto (≥{q3_lim:.1f}) Desempenho\nN_baixo = {n_baixo} | N_alto = {n_alto}',
        wrap=True, horizontalalignment='center', fontsize=8, style='italic'
    )

    # Estética final
    ax.yaxis.tick_right()
    ax.set(ylabel=None)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_position([0.22, 0.1, 0.7, 0.8])
    plt.tight_layout()

    # Salvar
    if salvar:
        salvar_figura(nome_arquivo=f'top{top_n}_diferencas_perfil', diretorio=diretorio, materia=materia)

    plt.show()

