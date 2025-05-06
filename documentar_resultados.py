# ======================================
# Módulo: documentar_resultados.py
# ======================================
"""
Estilização e exportação de resultados gráficos e tabulares.

Este módulo centraliza a identidade visual:
  - Configuração de paletas, estilos e colormaps.
  - Dimensionamento automático de figuras e ajuste de fontes.
  - Criação padronizada de figuras/axes.
  - Salvamento de gráficos em pastas organizadas.
  - Exportação de DataFrames para LaTeX.

Functions:
    aplicar_estilo_visual(paleta, retornar_cmap=False) → Union[List[str], LinearSegmentedColormap]
    dimensionar_figuresize(n_linhas=1, n_colunas=1, largura_base=6, altura_base=4.2,
                           largura_maxima=None, modo=None) → Tuple[float, float]
    ajustar_fontsize_por_figsize(figwidth, base_width=6) → Dict[str, float]
    padronizar_figura(linhas, colunas, new_args=None, include_args_subplot=False, escala=0.8) → Tuple[Figure, Any, Dict[str, float]]
    salvar_figura(nome_arquivo, materia, diretorio='figuras', formato='png') → None
"""



#--------------------------------------------------------------------------------------------------------------



import pandas as pd
import numpy as np

import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# SALVAMENTO E EXPORTAÇÃO
# -------------------------------------------------------------------------------


def salvar_figura(nome_arquivo, materia, diretorio='figuras', formato='png'):
    """
    Salva a figura atual do Matplotlib em diretório organizado.

    Args:
        nome_arquivo (str): Nome base do arquivo a ser salvo.
        materia (str): Nome da matéria ou categoria para compor o nome do arquivo.
        diretorio (str, optional): Nome da subpasta para armazenar as imagens. Default é 'figuras'.
        formato (str, optional): Formato do arquivo (e.g., 'png', 'jpg'). Default é 'png'.

    Returns:
        None
    """
    # Diretório centralizado para as imagens coletadas

    pasta_raiz = 'imagens'
    os.makedirs(pasta_raiz, exist_ok=True)

    # Define caminho da subpasta dentro de 'imagens'
    
    caminho_pasta = os.path.join(pasta_raiz, diretorio)
    os.makedirs(caminho_pasta, exist_ok=True)

    # Define caminho completo do arquivo
    
    if materia:
        nome_completo = f"{nome_arquivo}_{materia}.{formato}"
    else:
        nome_completo = f"{nome_arquivo}.{formato}"

    caminho_completo = os.path.join(caminho_pasta, nome_completo)

    # Figura salva
    
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    
    # Emissão de sinalização para indicar que a figura foi salva

    print(f"Figura salva em: {caminho_completo}")


#--------------------------------------------------------------------------------------------------------------

def exportar_df_para_latex(df, nome_tabela="tabela", caminho_pasta="./tables", index=False,
                            caption=None, label=None, ajustar_cabecalhos=True, limite=25):
    """
    Exporta um DataFrame pandas para LaTeX com suporte a nomes de colunas longos.

    Parâmetros:
        df (pandas.DataFrame): DataFrame a ser exportado.
        nome_tabela (str): Nome do arquivo .tex.
        caminho_pasta (str): Caminho onde salvar.
        index (bool): Se True, inclui o índice.
        caption (str): Legenda da tabela.
        label (str): Label para \ref.
        ajustar_cabecalhos (bool): Se True, aplica quebra de linhas nos nomes das colunas.
        limite (int): Número de caracteres por linha antes da quebra.

    Retorna:
        None
    """

    # Cria a pasta se necessário
    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)

    # Legenda e label automáticos se não fornecidos
    caption = caption or f"Tabela: {nome_tabela.replace('_', ' ').capitalize()}"
    label = label or f"tab:{nome_tabela.lower()}"
    caminho_arquivo = os.path.join(caminho_pasta, f"{nome_tabela}.tex")


    # Aplica quebra nos nomes das colunas, se desejado
    if ajustar_cabecalhos:
        colunas_ajustadas = quebrar_nomes_latex({col: col for col in df.columns}, limite)
        df = df.rename(columns=colunas_ajustadas)
    
    conteudo_tabela = df.to_latex(index=index,
                                   escape=False,
                                   longtable=False,
                                   multicolumn=True,
                                   multicolumn_format='c')

    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(f"""\\begin{{table}}[H]
                    \\centering
                    \\caption{{{caption}}}
                    \\label{{{label}}}
                    \\adjustbox{{max width=\\textwidth}}{{%
                    {conteudo_tabela.strip()}
                    }}
                    \\end{{table}}
                    % Para inserir esta tabela no texto:
                    % ----------------------------------
                    % \\input{{{os.path.join(caminho_pasta, nome_tabela + '.tex')}}}
                    """)

    print(f"Tabela salva com sucesso em: {caminho_arquivo}")