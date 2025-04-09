import pandas as pd
import numpy as np
 
def simular_mortalidade(populacao_path, tabuas_path, num_simulacoes=1000, anos_simulacao=5):
    """
    Simula a mortalidade em uma população ao longo de um período de anos.
 
    Args:
        populacao_path (str): Caminho para o arquivo Excel da população.
        tabuas_path (str): Caminho para o arquivo Excel das tábuas de mortalidade.
        num_simulacoes (int): Número de simulações a serem realizadas.
        anos_simulacao (int): Número de anos para cada simulação.
 
    Returns:
        pandas.DataFrame: DataFrame contendo os resultados das simulações.
    """
 
    # Carrega os dados da população e das tábuas de mortalidade
    populacao = pd.read_excel(populacao_path)
    tabuas = pd.read_excel(tabuas_path)
 
    # Cria um DataFrame para armazenar os resultados das simulações
    resultados = pd.DataFrame(index=range(num_simulacoes), columns=[f'mortes_ano_{i+1}' for i in range(anos_simulacao)] + ['mortes_total'])
 
    # Realiza as simulações
    for simulacao in range(num_simulacoes):
        populacao_atual = populacao.copy()
        mortes_total = 0
 
        for ano in range(anos_simulacao):
            mortes_ano = 0
            for idade in range(116):
                # Obtém a probabilidade de morte para cada sexo
                prob_morte_fem = tabuas.loc[idade, 'Feminino']
                prob_morte_masc = tabuas.loc[idade, 'Masculino']
 
                # Calcula o número de mortes para cada sexo, com aleatoriedade
                mortes_fem = np.random.binomial(populacao_atual.loc[idade, 'Feminino'], prob_morte_fem)
                mortes_masc = np.random.binomial(populacao_atual.loc[idade, 'Masculino'], prob_morte_masc)
 
                # Atualiza a população
                populacao_atual.loc[idade, 'Feminino'] -= mortes_fem
                populacao_atual.loc[idade, 'Masculino'] -= mortes_masc
 
                mortes_ano += mortes_fem + mortes_masc
 
            # Registra o número de mortes no ano
            resultados.loc[simulacao, f'mortes_ano_{ano+1}'] = mortes_ano
            mortes_total += mortes_ano
 
            # Envelhece a população
            populacao_atual.loc[1:115, ['Feminino', 'Masculino']] = populacao_atual.loc[0:114, ['Feminino', 'Masculino']].values
            populacao_atual.loc[0, ['Feminino', 'Masculino']] = 0
 
        # Registra o total de mortes na simulação
        resultados.loc[simulacao, 'mortes_total'] = mortes_total
 
    return resultados
 
# Exemplo de uso
populacao_path = 'populacao.xlsx'
tabuas_path = 'tabuas.xlsx'
resultados = simular_mortalidade(populacao_path, tabuas_path)
 
# Salva os resultados em um arquivo Excel
resultados.to_excel('resultados_simulacao.xlsx', index=False)
 
# Gera um relatório estatístico
relatorio = resultados.describe()
print(relatorio)