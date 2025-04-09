"""
Gerado pelo ChatGPT
"""
import time
import pandas as pd
import numpy as np

# Parâmetros
N_SIMULACOES = int(input("Número de simulações a executar (recomendado: 1000-10000): ") or 1000)
ANOS = 5

inicio = time.time()

# Leitura dos dados
populacao_df = pd.read_excel("populacao.xlsx")
tabua_df = pd.read_excel("tabuas.xlsx")

# Organização dos dados
idades = populacao_df['Idade']
pop_fem = populacao_df['Feminino'].astype(int).values
pop_masc = populacao_df['Masculino'].astype(int).values

'''print("Populações:")
print(pop_fem)
print(pop_masc)'''

qx_fem = tabua_df['Feminino'].values
qx_masc = tabua_df['Masculino'].values

max_idade = len(idades) - 1

# Inicializar armazenamento de resultados
obitos_anuais_fem = np.zeros((N_SIMULACOES, ANOS), dtype=int)
obitos_anuais_masc = np.zeros((N_SIMULACOES, ANOS), dtype=int)

prep = time.time()

for sim in range(N_SIMULACOES):
    # Copiar população para a simulação atual
    pop_f = pop_fem.copy()
    pop_m = pop_masc.copy()

    for ano in range(ANOS):
        mortes_f = np.random.binomial(pop_f, qx_fem)
        mortes_m = np.random.binomial(pop_m, qx_masc)

        obitos_anuais_fem[sim, ano] = mortes_f.sum()
        obitos_anuais_masc[sim, ano] = mortes_m.sum()

        '''print(f"Simulação {sim}, ano {ano}, mortes:")
        print(mortes_f)
        print(mortes_m)'''

        # Atualizar população: remove mortos
        pop_f -= mortes_f
        pop_m -= mortes_m

        '''print("Populações após mortes:")
        print(pop_f)
        print(pop_m)'''

        # Envelhecer: desloca população para frente
        pop_f = np.roll(pop_f, 1)
        pop_f[0] = 0  # sem nascimentos
        pop_f[max_idade] = 0  # todos que chegam à idade máxima morrem no próximo ano

        pop_m = np.roll(pop_m, 1)
        pop_m[0] = 0
        pop_m[max_idade] = 0

        '''print("Populações após envelhecimento:")
        print(pop_f)
        print(pop_m)'''

fim_simulacoes = time.time()

print(f"Tempo de execução das simulações: {(fim_simulacoes - prep):.2f} segundos")

# Cálculo de estatísticas
total_obitos_fem = obitos_anuais_fem.sum(axis=1)
total_obitos_masc = obitos_anuais_masc.sum(axis=1)
total_obitos = total_obitos_fem + total_obitos_masc

relatorio = pd.DataFrame({
    "Óbitos Fem. Total": total_obitos_fem,
    "Óbitos Masc. Total": total_obitos_masc,
    "Óbitos Total": total_obitos
})

estatisticas = relatorio.describe()

# Salvar em Excel
with pd.ExcelWriter("resultado_simulacoes.xlsx") as writer:
    relatorio.to_excel(writer, sheet_name="Simulações", index=False)
    estatisticas.to_excel(writer, sheet_name="Resumo Estatístico")

fim = time.time()

print(f"Tempo total: {(fim - inicio):.2f} segundos")

print("Simulações concluídas. Resultados salvos em 'resultado_simulacoes.xlsx'")
