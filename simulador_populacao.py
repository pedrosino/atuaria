""" Script para simular o comportamento de uma população ao longo de determinado tempo,
comparando com uma tábua de mortalidade.                                           
                                                                             
Desenvolvido por Pedro Santos Guimarães - abril de 2025   """

import time
import math
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Log simples')
# Parâmetros da simulação
TAMANHO_POPULACAO = 3000
NUMERO_SIMULACOES = 100000
IDADE_MINIMA = 40
IDADE_MAXIMA = 80
PROPORCAO_MASCULINO = 0.6

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
tabuas = pd.read_excel("tabuas.xlsx")

# Inicialmente estava usando apenas uma tábua para toda a população
tabua = tabuas[tabuas.columns[1]].to_numpy()

# Salva as duas tábuas na mesma variável - verificar se é o melhor
# Talvez se não transpuser fique mais fácil?
tabuas_juntas = tabuas[tabuas.columns[1:]].to_numpy().T

#logging.info(tabuas_juntas)

###populacao_original = np.random.randint(IDADE_MINIMA, IDADE_MAXIMA, TAMANHO_POPULACAO)
populacao_idades = np.random.randint(IDADE_MINIMA, IDADE_MAXIMA, TAMANHO_POPULACAO)
aleatorios_sexo = np.random.random(TAMANHO_POPULACAO)
populacao_sexo = (aleatorios_sexo < PROPORCAO_MASCULINO).astype(int)
populacao_original = [populacao_idades, populacao_sexo]
print("População:")
print(populacao_original)
###esperado_original = tabua[populacao_original]
esperado_original = tabuas_juntas[populacao_original[1], populacao_original[0]]
total_esperado = esperado_original.sum()
print(esperado_original)
print(f"Esperados: {total_esperado}")

desvio = math.sqrt(total_esperado)
margem = desvio*4
maximo = total_esperado + margem
print(f"Máximo: {maximo}")

esperados = np.zeros((NUMERO_SIMULACOES,), dtype=float)

mortes_ano = np.zeros((NUMERO_SIMULACOES,5), dtype=int)

morreram = np.zeros((int(maximo)*5,), dtype=int)

mortes = np.empty((5, TAMANHO_POPULACAO))

novos_esperados = np.empty((5, TAMANHO_POPULACAO))

input("Iniciar simulações")

start_time = time.time()

populacao = np.empty_like(populacao_original)
esperado = np.empty_like(esperado_original)

for pop in range(NUMERO_SIMULACOES):
    populacao[:] = populacao_original #= populacao_original.copy()
    esperado[:] = esperado_original #= esperado_original.copy()

    total_pop = 0

    for ano in range(5):
        aleatorios = np.random.random(TAMANHO_POPULACAO)

        morreu = (aleatorios < esperado).astype(int)
        #print(f"..Simulação {pop}, ano {ano}..")
        #print("Sorteio:")
        #print(aleatorios[-300:])
        #print(f"Morreram {morreu.sum()}")
        #print(morreu[-100:])
        mortes[ano] = morreu

        # adiciona 1 ano a cada pessoa
        ###populacao = np.minimum(populacao +1, 116)
        populacao[0] = np.minimum(populacao[0] + 1, 116)

        #print(populacao[:][-100:])

        # atualiza as probabilidades
        ###esperado = tabua[populacao]
        esperado = tabuas_juntas[populacao[1], populacao[0]] #tabua[populacao]

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

arquivo = f"Pop_aleatoria_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"

# create a excel writer object
with pd.ExcelWriter(arquivo) as writer:

    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    df_pop = pd.DataFrame({'Idade': populacao_original[0], 'Sexo': populacao_original[1], 'Esperado': esperado_original}) #populacao_original, columns=['Idade'])
    df_pop.to_excel(writer, sheet_name="Populacao", index=False)

    df_morreram = pd.DataFrame(morreram, columns=['Quantidade'])
    df_coluna = df_morreram.reset_index()
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
