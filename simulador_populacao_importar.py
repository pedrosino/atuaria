""" Script para simular o comportamento de uma população ao longo de determinado tempo,
comparando com uma tábua de mortalidade.                                           
                                                                             
Desenvolvido por Pedro Santos Guimarães - abril de 2025   """

import time
import math
import numpy as np
import pandas as pd

# Parâmetros da simulação
NUMERO_SIMULACOES = 1000

np.set_printoptions(linewidth=76)

'''
=SEERRO(
  MÍNIMO(
    115;
    MÁXIMO(
        0;
        ARRED(
          INV.NORM.N(
            ALEATÓRIO();
            ÍNDICE($K$8:$K$18;CORRESP(A2;$M$8:$M$18;-1));
            ÍNDICE($L$8:$L$18;CORRESP(A2;$M$8:$M$18;-1)));
          0)
        )
      );
  "--")
'''

# Ler tábua de mortalidade do excel
tabuas = pd.read_excel("tabuas116.xlsx")

# Inicialmente estava usando apenas uma tábua para toda a população
tabua = tabuas[tabuas.columns[1]].to_numpy()

# Salva as duas tábuas na mesma variável - verificar se é o melhor
# Talvez se não transpuser fique mais fácil?
tabuas_juntas = tabuas[tabuas.columns[1:]].to_numpy().T

# Lê população do arquivo excel
arq_pop = pd.read_excel("pop_pbd.xlsx")
qtde_pop = arq_pop[arq_pop.columns[1:]].to_numpy().T

idades = np.arange(116)

###populacao_original = np.repeat(idades, qtde_pop)
idades_feminina = np.repeat(idades, qtde_pop[0])
idades_masculina =  np.repeat(idades, qtde_pop[1])
populacao_feminina = [idades_feminina, np.zeros((idades_feminina.size,), dtype=int)]
populacao_masculina = [idades_masculina, np.ones((idades_masculina.size,), dtype=int)]

populacao_original = np.concatenate([populacao_feminina, populacao_masculina], axis=1)
tamanho_populacao = populacao_original[0].size

print("População:")
print(tamanho_populacao)
print(populacao_original[:, -100:])
### esperado_original = tabua[populacao_original]
esperado_original = tabuas_juntas[populacao_original[1], populacao_original[0]]
total_esperado = esperado_original.sum()
print(esperado_original[-100:])
print(f"Esperados: {total_esperado}")

desvio = math.sqrt(total_esperado)
margem = desvio*4
maximo = total_esperado + margem
print(f"Máximo: {maximo}")

esperados = np.zeros((NUMERO_SIMULACOES,), dtype=float)

mortes_ano = np.zeros((NUMERO_SIMULACOES,5), dtype=int)

morreram = np.zeros((int(maximo)*6,), dtype=int)

mortes = np.empty((5, tamanho_populacao))

novos_esperados = np.empty((5, tamanho_populacao))

input("Iniciar simulações")

start_time = time.time()

populacao = np.empty_like(populacao_original)
esperado = np.empty_like(esperado_original)

for pop in range(NUMERO_SIMULACOES):
    populacao[:] = populacao_original #= populacao_original.copy()
    esperado[:] = esperado_original #= esperado_original.copy()

    total_pop = 0

    for ano in range(5):
        # Gera números aleatórios para simular a morte ou não
        aleatorios = np.random.random(tamanho_populacao)

        morreu = (aleatorios < esperado).astype(int)
        #print(f"..Simulação {pop}, ano {ano}..")
        #print("Sorteio:")
        #print(aleatorios[-100:])
        #print(f"Morreram {morreu.sum()}")
        #print(morreu[-100:])
        mortes[ano] = morreu

        # adiciona 1 ano a cada pessoa
        ###populacao = np.minimum(populacao + 1, 116)
        populacao[0] = np.minimum(populacao[0] + 1, 116)

        #print(populacao[:, -100:])

        # atualiza as probabilidades
        ###esperado = tabua[populacao]
        esperado = tabuas_juntas[populacao[1], populacao[0]]

        # as que morreram passam a ter probabilidade zero
        esperado[morreu == 1] = 0
        ###populacao[morreu == 1] = 116
        populacao[0][morreu == 1] = 116

        novos_esperados[ano] = esperado

        #print(f"Sobraram {np.count_nonzero(esperado)}")
        #print(f"Novo esperado: {esperado.sum()}")

        #print("Ficou")
        #print(esperado[-100:])
        #input("...")

        total_ano = morreu.sum()
        mortes_ano[pop][ano] = total_ano
        total_pop += total_ano

    #print(f"Simulação {pop+1:4d}. Real: {mortes_ano[pop]}, total {total_pop}")

    if (pop+1)%(NUMERO_SIMULACOES/100) == 0:
        print(f"Simulação {pop+1:4d}. Real: {mortes_ano[pop]}, total {total_pop}")
    #    input("Continuar...")

    morreram[total_pop] += 1

print(morreram)

arquivo = f"Pop_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"

# create a excel writer object
with pd.ExcelWriter(arquivo) as writer:

    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    ##df_pop = pd.DataFrame({'Idade': populacao_original, 'Esperado': esperado_original}) #populacao_original, columns=['Idade'])
    df_pop = pd.DataFrame({'Idade': populacao_original[0], 'Sexo': populacao_original[1], 'Esperado': esperado_original}) #populacao_original, columns=['Idade'])
    df_pop.to_excel(writer, sheet_name="Populacao", index=False)

    df_morreram = pd.DataFrame(morreram, columns=['Quantidade'])
    df_coluna = df_morreram.reset_index()#df_morreram[df_morreram['Quantidade'] != 0].reset_index()
    df_coluna.columns = ['Ocorridos', 'Quantidade']
    df_coluna.to_excel(writer, sheet_name="Morreram", index=False)

    df_mortes = pd.DataFrame(mortes).T
    df_mortes.columns=['Ano1','Ano2','Ano3','Ano4','Ano5']
    df_mortes.to_excel(writer, sheet_name="Mortes", index=False)

    df_esperados = pd.DataFrame(novos_esperados).T#, columns=['Ano1','Ano2','Ano3','Ano4','Ano5'])
    df_esperados.columns=['Ano1','Ano2','Ano3','Ano4','Ano5']
    df_esperados.to_excel(writer, sheet_name="Esperados", index=False)

print(f"Resultados salvos no arquivo {arquivo}")
print(f"... {(time.time() - start_time)} segundos ...")
