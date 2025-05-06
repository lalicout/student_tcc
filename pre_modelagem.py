# ======================================
# Módulo: pre_modelagem.py
# ======================================
"""
Pré-modelagem de dados de desempenho escolar.

Este módulo implementa as etapas de importação, limpeza, codificação,
(escalonamento opcional) e balanceamento de classes das bases de dados
de estudantes para tarefas de classificação ou regressão.

Functions:
    importar_base(materia: str) → pd.DataFrame
    preparar_dados(df: pd.DataFrame, scaling: bool = False, classificacao: bool = True) → pd.DataFrame
    balancear_dados(X: array-like, y: array-like) → tuple
"""

#Bibliotecas

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from IPython.display import display



def importar_base(materia):
    """
    Lê o CSV de estudantes, traduz colunas e valores, e adiciona coluna de aprovação.

    Args:
        materia (str):
            Código da disciplina —
            'mat' para Matemática ou 'por' para Português.

    Returns:
        pd.DataFrame:
            DataFrame com colunas renomeadas, valores traduzidos
            e nova coluna 'aprovacao' (0/1).

    Raises:
        ValueError: Se `materia` não for 'mat' nem 'por'.
    """

    # Tratamento de exceção para verificar se a matéria é válida
    
    if materia not in ['mat', 'por']:
        if materia == 'portugues':
            materia = 'por'
        
        elif materia == 'matematica':
            materia = 'mat'
        else:
            raise ValueError("O parâmetro 'materia' deve ser 'mat' para Matemática ou 'por' para Português.")
    
    # Cria o caminho completo do arquivo CSV de forma segura e dinâmica, baseado na matéria escolhida
 
    base_path = os.path.join(os.path.expanduser("~"), "student_performance_tcc", "data")
    arquivo = f"student-{materia}.csv"
    caminho_completo = os.path.join(base_path, arquivo)
    
    # Leitura do arquivo CSV correspondente à matéria, com separador ';'

    df = pd.read_csv(caminho_completo,sep=';')

    #Dicionário de traduções de nomes de colunas
    
    colunas_renomeadas = {
        'school': 'escola',
        'sex': 'genero',
        'age': 'idade',
        'address': 'endereco',
        'famsize': 'tamanho_familia',
        'Pstatus': 'status_parental',  # Estado civil ou convivência dos pais
        'Medu': 'escolaridade_mae',
        'Fedu': 'escolaridade_pai',
        'Mjob': 'profissao_mae',
        'Fjob': 'profissao_pai',
        'reason': 'motivo_escolha_escola',
        'guardian': 'responsavel_legal',
        'traveltime': 'tempo_transporte',
        'studytime': 'tempo_estudo',
        'failures': 'reprovacoes',  
        'schoolsup': 'apoio_escolar',
        'famsup': 'apoio_familiar',
        'paid': 'aulas_particulares',
        'activities': 'atividades_extracurriculares',
        'nursery': 'frequentou_creche',
        'higher': 'intencao_superior',  # Reflete o desejo do aluno de prosseguir para o ensino superior
        'internet': 'acesso_internet',
        'romantic': 'relacionamento_romantico',
        'famrel': 'relacao_familiar',
        'freetime': 'tempo_livre',
        'goout': 'frequencia_saidas',
        'Dalc': 'alcool_dias_uteis',
        'Walc': 'alcool_fim_semana',
        'health': 'saude',
        'absences': 'faltas',
        'G1': 'nota1',
        'G2': 'nota2',
        'G3': 'nota_final'
        }
    
    # Dicionário de mapeamentos para traduzir os valores das variáveis categóricas após o renomeio das colunas
    
    substituicoes = {
        'escola': {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
        'genero': {'F': 'Mulher', 'M': 'Homem'},
        'endereco': {'U': 'Urbano', 'R': 'Rural'},
        'tamanho_familia': {'GT3': 'Mais de 3 membros', 'LE3': '3 membros ou menos'},
        'status_parental': {'A': 'Separados', 'T': 'Juntos'},
        'profissao_mae': {'at_home': 'Dona de casa', 'health': 'Área da saúde', 'other': 'Outra profissão', 'services': 'Serviços', 'teacher': 'Professor(a)'},
        'profissao_pai': {'at_home': 'Dono de casa', 'health': 'Área da saúde', 'other': 'Outra profissão', 'services': 'Serviços', 'teacher': 'Professor(a)'},
        'motivo_escolha_escola': {'course': 'Curso específico', 'other': 'Outro motivo', 'home': 'Próximo de casa', 'reputation': 'Reputação da escola'},
        'responsavel_legal': {'mother': 'Mãe', 'father': 'Pai', 'other': 'Outro responsável'},
        'apoio_escolar': {'yes': 'Sim', 'no': 'Não'},
        'apoio_familiar': {'yes': 'Sim', 'no': 'Não'},
        'aulas_particulares': {'yes': 'Sim', 'no': 'Não'},
        'atividades_extracurriculares': {'yes': 'Sim', 'no': 'Não'},
        'frequentou_creche': {'yes': 'Sim', 'no': 'Não'},
        'intencao_superior': {'yes': 'Sim', 'no': 'Não'},
        'acesso_internet': {'yes': 'Sim', 'no': 'Não'},
        'relacionamento_romantico': {'yes': 'Sim', 'no': 'Não'}
    }
  
    # Renomeia colunas
    df.rename(columns=colunas_renomeadas, inplace=True)

    # Renomeia valores pelas substituições
    
    for coluna, variavel in substituicoes.items():
        df[coluna].replace(variavel, inplace=True)
    
    # Adiciona a variável 'aprovacao' com base na nota final
    df['aprovacao'] = df['nota_final'].apply(lambda x: 'Aprovado' if x >= 10 else 'Reprovado')

    return df

# -------------------------------------------------------------------------------------------------------------------


def preparar_dados(df, scaling=False, classificacao=True):
    """
    Codifica, remove colunas irrelevantes e (opcionalmente) escala as variáveis.

    Args:
        df (pd.DataFrame):
            DataFrame original vindo de `importar_base`.
        scaling (bool, optional):
            Se True, normaliza variáveis numéricas com StandardScaler. Default False.
        classificacao (bool, optional):
            Se True, prepara X para tarefa de classificação (remove G1, G2, G3).
            Caso contrário, remove coluna 'aprovacao'. Default True.

    Returns:
        pd.DataFrame:
            DataFrame pronto para modelagem, contendo variáveis
            codificadas e sem valores faltantes.

    Raises:
        KeyError: Se faltar alguma coluna obrigatória em `df`.
    """


    # Mapeamento de variáveis binárias (Label Encoding)
    df['tamanho_familia'] = df['tamanho_familia'].map({'Mais de 3 membros': 1, '3 membros ou menos': 0}).fillna(0)
    # Só mapeia se os valores forem strings
    
    df['aprovacao'] = df['aprovacao'].map({'Aprovado': 1, 'Reprovado': 0})
    print(df['aprovacao'].value_counts())
    
    colunas_label_encoding = [
        'apoio_escolar', 'apoio_familiar', 'aulas_particulares',
        'atividades_extracurriculares', 'frequentou_creche',
        'intencao_superior', 'acesso_internet', 'relacionamento_romantico'
    ]

    for coluna in colunas_label_encoding:
        df[coluna] = df[coluna].map({'Sim': 1, 'Não': 0})

    # Colunas categóricas nominais para One-Hot Encoding
    colunas_onehot = [
        'escola', 'genero', 'endereco', 'status_parental',
        'profissao_mae', 'profissao_pai', 'motivo_escolha_escola',
        'responsavel_legal'
    ]

    # Aplica One-Hot Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[colunas_onehot])
    column_names = onehot_encoder.get_feature_names_out(colunas_onehot)
    onehot_df = pd.DataFrame(onehot_encoded, columns=column_names, index=df.index)

    # Remove colunas originais que já foram codificadas
    df = df.drop(columns=colunas_onehot, axis=1)

    # Remove colunas irrelevantes para o tipo de modelo
    if classificacao:
        df = df.drop(columns=['nota1', 'nota2', 'nota_final'], axis=1)
        print(df['aprovacao'].value_counts())
    else:
        df = df.drop(columns=['aprovacao'], axis=1)

    # Aplica scaling, se solicitado
    if scaling:
        colunas_numericas = [
            'idade', 'escolaridade_mae', 'escolaridade_pai', 'tempo_transporte', 'tempo_estudo',
            'reprovacoes', 'relacao_familiar', 'tempo_livre', 'frequencia_saidas', 
            'alcool_dias_uteis', 'alcool_fim_semana', 'saude', 'faltas'
        ]

        # Imputação (evita drop de linhas)
        df[colunas_numericas] = df[colunas_numericas].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='mean')
        df[colunas_numericas] = imputer.fit_transform(df[colunas_numericas])

        # Escalonamento
        scaler = StandardScaler()
        df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

    # Concatena com one-hot
    df_final = pd.concat([df, onehot_df], axis=1)

    # Limpeza final
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()

    return df_final




def balancear_dados(X, y):
    """
    Aplica SMOTE-Tomek para balancear classes minoritárias.

    1) Converte X e y para numpy arrays;
    2) Resample;
    3) Reconstrói DataFrame/Series se necessário.

    Args:
        X (pd.DataFrame or np.ndarray): Matriz de features.
        y (pd.Series or np.ndarray): Vetor de labels binários.

    Returns:
        tuple:
            X_res (np.ndarray or pd.DataFrame): Features reamostradas.
            y_res (np.ndarray or pd.Series): Labels reamostrados.
    """
    import numpy as np
    import pandas as pd
    from imblearn.combine import SMOTETomek

    # Guarda info para reconstrução
    is_df = isinstance(X, pd.DataFrame)
    cols = X.columns if is_df else None
    y_name = y.name if hasattr(y, 'name') else None

    # Converte para numpy
    X_np = X.values if is_df else np.asarray(X)
    y_np = np.asarray(y)

    smt = SMOTETomek(random_state=42)
    X_res_np, y_res_np = smt.fit_resample(X_np, y_np)

    # Reconstrói, se for o caso
    if is_df:
        X_res = pd.DataFrame(X_res_np, columns=cols)
        y_res = pd.Series(y_res_np, name=y_name)
    else:
        X_res, y_res = X_res_np, y_res_np

    return X_res, y_res





# -------------------------------------------------------------------------------------------------------------------


# ======================================
# Fim do módulo
# ======================================
