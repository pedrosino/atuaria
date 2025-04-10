""" Script para simular o comportamento de uma população ao longo de determinado tempo,
comparando com uma tábua de mortalidade.                                           
                                                                             
Desenvolvido por Pedro Santos Guimarães - abril de 2025"""

import time
import numpy as np
import pandas as pd

# Parâmetros da simulação
NUMERO_SIMULACOES = int(input("Digite o número de simulações: ")) or 100
NUMERO_ANOS = int(input("Digite o número de anos em cada simulação: ")) or 5

# Ler tábua de mortalidade do excel
tabuas = pd.read_excel("tabuas.xlsx")

# Idades
idades = tabuas[tabuas.columns[0]].to_numpy()

# Salva as duas tábuas separadas
tabua_fem = tabuas[tabuas.columns[1]].to_numpy()
tabua_masc = tabuas[tabuas.columns[2]].to_numpy()

tamanho_populacao = len(idades)
idade_maxima = tamanho_populacao - 1

# Lê população do arquivo excel
arq_pop = pd.read_excel("populacao.xlsx")

# Salva as duas populações separadas
populacao_feminina_original = arq_pop[arq_pop.columns[1]].to_numpy(dtype=int)
populacao_masculina_original = arq_pop[arq_pop.columns[2]].to_numpy(dtype=int)

print(f"População inicial: {populacao_feminina_original.sum()} mulheres e " \
      f"{populacao_masculina_original.sum()} homens (total " \
      f"{populacao_feminina_original.sum() + populacao_masculina_original.sum()})")

# Calcula esperados iniciais
esperado_inicial_fem = populacao_feminina_original * tabua_fem
esperado_inicial_masc = populacao_masculina_original * tabua_masc

esp_fem = esperado_inicial_fem.sum()
esp_masc = esperado_inicial_masc.sum()
total_esperado = esp_fem + esp_masc

print("Esperados:")
print(f"Feminina: {esp_fem:.4f}, masculina: {esp_masc:.4f}, total: {total_esperado:.4f}")

# Arrays para registrar os números por simulação e ano
esperados_ano_fem = np.zeros((NUMERO_SIMULACOES, NUMERO_ANOS), dtype=float)
esperados_ano_masc = np.zeros((NUMERO_SIMULACOES, NUMERO_ANOS), dtype=float)

mortes_ano_fem = np.zeros((NUMERO_SIMULACOES, NUMERO_ANOS), dtype=int)
mortes_ano_masc = np.zeros((NUMERO_SIMULACOES, NUMERO_ANOS), dtype=int)

print("Iniciando simulações...")

start_time = time.time()

populacao_fem = np.empty_like(populacao_feminina_original)
populacao_masc = np.empty_like(populacao_masculina_original)

esperado_fem = np.empty_like(esperado_inicial_fem)
esperado_masc = np.empty_like(esperado_inicial_masc)

for sim in range(NUMERO_SIMULACOES):
    # Copia populações originais
    populacao_fem[:] = populacao_feminina_original.copy()
    populacao_masc[:] = populacao_masculina_original.copy()

    for ano in range(5):
        # Calcula esperados
        esperado_fem = populacao_fem * tabua_fem
        esperado_masc = populacao_masc * tabua_masc

        esperados_ano_fem[sim, ano] = esperado_fem.sum()
        esperados_ano_masc[sim, ano] = esperado_masc.sum()

        # Simula as mortes em cada idade
        morreram_fem = np.random.binomial(populacao_fem, tabua_fem)
        morreram_masc = np.random.binomial(populacao_masc, tabua_masc)

        mortes_ano_fem[sim, ano] = morreram_fem.sum()
        mortes_ano_masc[sim, ano] = morreram_masc.sum()

        # Remove os que morreram
        populacao_fem -= morreram_fem
        populacao_masc -= morreram_masc

        # Adiciona 1 ano a cada pessoa
        populacao_fem = np.roll(populacao_fem, 1)
        populacao_fem[0] = 0
        populacao_fem[idade_maxima] = 0

        populacao_masc = np.roll(populacao_masc, 1)
        populacao_masc[0] = 0
        populacao_masc[idade_maxima] = 0

    if (sim+1)%(NUMERO_SIMULACOES/100) == 0:
        print(f"Simulação {sim+1:4d} concluída.")

print(f"... Simulações finalizadas em {(time.time() - start_time)} segundos ...")

# Prepara as variáveis para o excel

# Colunas para a planilha
fem_cols = [f"Fem_Ano{i+1}" for i in range(NUMERO_ANOS)]
masc_cols = [f"Masc_Ano{i+1}" for i in range(NUMERO_ANOS)]
total_cols = [f"Total_Ano{i+1}" for i in range(NUMERO_ANOS)]
soma_fem_col = ["Fem_Total"]
soma_masc_col = ["Masc_Total"]
soma_geral_col = ["Total_Geral"]

# Junta arrays de mortes
mortes_ano_total = mortes_ano_fem + mortes_ano_masc
total_mortes_fem = mortes_ano_fem.sum(axis=1).reshape(-1, 1)
total_mortes_masc = mortes_ano_masc.sum(axis=1).reshape(-1, 1)
total_mortes = mortes_ano_total.sum(axis=1).reshape(-1, 1)

todas_cols = fem_cols + masc_cols + total_cols + soma_fem_col + soma_masc_col + soma_geral_col
mortes_junto = np.hstack((
    mortes_ano_fem, mortes_ano_masc, mortes_ano_total,
    total_mortes_fem, total_mortes_masc, total_mortes))
df_mortes = pd.DataFrame(mortes_junto, columns=todas_cols)

# Junta arrays de esperados
esperados_ano_total = esperados_ano_fem + esperados_ano_masc
total_esperados_fem = esperados_ano_fem.sum(axis=1).reshape(-1, 1)
total_esperados_masc = esperados_ano_masc.sum(axis=1).reshape(-1, 1)
total_esperados = esperados_ano_total.sum(axis=1).reshape(-1, 1)

esperados_junto = np.hstack((
    esperados_ano_fem, esperados_ano_masc, esperados_ano_total,
    total_esperados_fem, total_esperados_masc, total_esperados))
df_esperados = pd.DataFrame(esperados_junto, columns=todas_cols)

# Contadores
unicos_fem, count_fem = np.unique_counts(total_mortes_fem)
unicos_masc, count_masc = np.unique_counts(total_mortes_masc)
unicos_geral, count_geral = np.unique_counts(total_mortes)

df_fem = pd.DataFrame({"Fem_Unicos": unicos_fem, "Qtde_Fem": count_fem})
df_masc = pd.DataFrame({"Masc_Unicos": unicos_masc, "Qtde_Masc": count_masc})
df_geral = pd.DataFrame({"Geral_Unicos": unicos_geral, "Qtde_Geral": count_geral})

df_contadores = pd.concat([df_fem, df_masc, df_geral], axis=1)

# Estatísticas
estatisticas_mortes = df_mortes.describe()
estatisticas_esperados = df_esperados.describe()

arquivo = f"Simulação_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"

# create a excel writer object
with pd.ExcelWriter(arquivo) as writer:
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet

    arq_pop.to_excel(writer, sheet_name="Populacao", index=False)
    tabuas.to_excel(writer, sheet_name="Tábua", index=False)
    df_mortes.to_excel(writer, sheet_name="Mortes", index=False)
    df_esperados.to_excel(writer, sheet_name="Esperados", index=False)
    estatisticas_mortes.to_excel(writer, sheet_name="Estatisticas_Mortes")
    estatisticas_esperados.to_excel(writer, sheet_name="Estatisticas_Esperados")
    df_contadores.to_excel(writer, sheet_name="Contadores", index=False)

print(f"Resultados salvos no arquivo {arquivo}")
print(f"...Tempo total: {(time.time() - start_time)} segundos ...")
