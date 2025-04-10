"""Script para realizar simulações de mortalidade em uma população, conforme uma tábua biométrica.
Gerado pelo Claude IA - versão 0.4"""

import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimulacaoMortalidade:
    """Classe principal"""
    def __init__(self, arquivo_populacao, arquivo_tabuas):
        # Carrega os dados iniciais
        self.populacao_original = pd.read_excel(arquivo_populacao)
        self.tabuas = pd.read_excel(arquivo_tabuas)

        # Renomeia as colunas para facilitar
        self.populacao_original.columns = ['idade', 'feminino', 'masculino']
        self.tabuas.columns = ['idade', 'prob_morte_fem', 'prob_morte_masc']

        # Verifica se os dados estão completos
        self.idade_maxima = 115  # Conforme especificado
        self.verificar_dados()

        # Pré-processa os dados para melhor desempenho
        self.otimizar_estruturas_dados()

        # Calcula os óbitos esperados (determinísticos) apenas para o primeiro ano
        self.calcular_obitos_esperados_inicial()

    def verificar_dados(self):
        """Verifica se os dados de entrada estão completos e consistentes"""
        # Verifica se todas as idades de 0 a 115 estão presentes
        idades_esperadas = set(range(self.idade_maxima + 1))
        idades_populacao = set(self.populacao_original['idade'])
        idades_tabuas = set(self.tabuas['idade'])

        if idades_esperadas != idades_populacao:
            raise ValueError(f"Arquivo de população não contém todas as idades de 0 a {self.idade_maxima}")

        if idades_esperadas != idades_tabuas:
            raise ValueError(f"Arquivo de tábuas não contém todas as idades de 0 a {self.idade_maxima}")

    def otimizar_estruturas_dados(self):
        """Pré-processa os dados para acesso mais rápido durante as simulações"""
        # Converte para arrays numpy para operações mais rápidas
        self.pop_fem_array = np.array(self.populacao_original['feminino'], dtype=int)
        self.pop_masc_array = np.array(self.populacao_original['masculino'], dtype=int)
        #print(f"Fem: {self.pop_fem_array.dtype}")
        #print(f"Masc: {self.pop_masc_array.dtype}")
        #input('...')

        # Cria dicionários de probabilidades por idade para acesso rápido
        self.prob_morte_fem = {}
        self.prob_morte_masc = {}

        for _, row in self.tabuas.iterrows():
            idade = int(row['idade'])
            self.prob_morte_fem[idade] = row['prob_morte_fem']
            self.prob_morte_masc[idade] = row['prob_morte_masc']

        # Probabilidades em np.array - muito mais rápido
        self.prob_fem = np.array(list(self.prob_morte_fem.values()))
        self.prob_masc = np.array(list(self.prob_morte_masc.values()))

    def calcular_obitos_esperados_inicial(self):
        """Calcula os óbitos esperados (determinísticos) apenas para o primeiro ano"""
        # Arrays para armazenar os resultados
        self.obitos_esperados_inicial = {}

        # Copia da população inicial
        pop_fem = self.pop_fem_array.copy()
        pop_masc = self.pop_masc_array.copy()

        # Calcula os óbitos esperados para cada idade
        obitos_fem_esperados = np.zeros_like(pop_fem, dtype=float)
        obitos_masc_esperados = np.zeros_like(pop_masc, dtype=float)

        for idade in range(self.idade_maxima + 1):
            # Calcula os óbitos esperados (determinístico - valor médio)
            obitos_fem_esperados[idade] = pop_fem[idade] * self.prob_morte_fem[idade]
            obitos_masc_esperados[idade] = pop_masc[idade] * self.prob_morte_masc[idade]

        # Calcula os óbitos esperados
       # obitos_fem_esperados = pop_fem * self.prob_fem
       # obitos_masc_esperados = pop_masc * self.prob_masc

        # Total de óbitos esperados para o primeiro ano
        self.obitos_esperados_inicial = {
            'ano': 1,
            'total': np.sum(obitos_fem_esperados) + np.sum(obitos_masc_esperados),
            'feminino': np.sum(obitos_fem_esperados),
            'masculino': np.sum(obitos_masc_esperados),
            'por_idade_fem': obitos_fem_esperados.copy(),
            'por_idade_masc': obitos_masc_esperados.copy()
        }

    def calcular_obitos_esperados_para_populacao(self, pop_fem, pop_masc, ano):
        """Calcula os óbitos esperados para uma dada população"""
        # Calcula os óbitos esperados para cada idade
        obitos_fem_esperados = np.zeros_like(pop_fem, dtype=float)
        obitos_masc_esperados = np.zeros_like(pop_masc, dtype=float)

        for idade in range(self.idade_maxima + 1):
            # Calcula os óbitos esperados (determinístico - valor médio)
            obitos_fem_esperados[idade] = pop_fem[idade] * self.prob_morte_fem[idade]
            obitos_masc_esperados[idade] = pop_masc[idade] * self.prob_morte_masc[idade]

        # Calcula os óbitos esperados
       # obitos_fem_esperados = pop_fem * self.prob_fem
       # obitos_masc_esperados = pop_masc * self.prob_masc

        # Total de óbitos esperados para este ano
        return {
            'ano': ano,
            'total': np.sum(obitos_fem_esperados) + np.sum(obitos_masc_esperados),
            'feminino': np.sum(obitos_fem_esperados),
            'masculino': np.sum(obitos_masc_esperados),
            'por_idade_fem': obitos_fem_esperados.copy(),
            'por_idade_masc': obitos_masc_esperados.copy()
        }

    def get_obitos_esperados_inicial_df(self):
        """Retorna um DataFrame com os óbitos esperados iniciais"""
        return pd.DataFrame([{
            'ano': self.obitos_esperados_inicial['ano'],
            'obitos_esperados_total': self.obitos_esperados_inicial['total'],
            'obitos_esperados_feminino': self.obitos_esperados_inicial['feminino'],
            'obitos_esperados_masculino': self.obitos_esperados_inicial['masculino']
        }])

    def simular_periodo(self, anos=5, seed=None):
        """Realiza uma simulação de mortalidade para um período de anos"""
        # Define seed para reprodutibilidade se fornecido
        if seed is not None:
            np.random.seed(seed)

        # Copia os arrays de população para não alterar os originais
        pop_fem = self.pop_fem_array.copy()
        pop_masc = self.pop_masc_array.copy()

        # Inicializa o contador de óbitos por ano
        obitos_anuais = []
        obitos_esperados_anuais = []
        populacao_anual = []

        # Registra população inicial
        pop_inicial = {
            'ano': 0, 
            'total': pop_fem.sum() + pop_masc.sum(),
            'feminino': pop_fem.sum(), 
            'masculino': pop_masc.sum()
        }
        populacao_anual.append(pop_inicial)

        for ano in range(1, anos + 1):
            # 1. Calcular óbitos esperados para este ano com base na população atual
            esperados = self.calcular_obitos_esperados_para_populacao(pop_fem, pop_masc, ano)
            obitos_esperados_anuais.append({
                'ano': ano,
                'total': esperados['total'],
                'feminino': esperados['feminino'],
                'masculino': esperados['masculino']
            })

            # 2. Simular óbitos reais para este ano
            # Inicializa arrays para óbitos
            obitos_fem = np.zeros_like(pop_fem)
            obitos_masc = np.zeros_like(pop_masc)

            # Calcula os óbitos para cada idade usando vetorização numpy
            for idade in range(self.idade_maxima + 1):
                # Simula os óbitos usando distribuição binomial
                if pop_fem[idade] > 0:
                    obitos_fem[idade] = np.random.binomial(int(pop_fem[idade]), self.prob_morte_fem[idade])
                if pop_masc[idade] > 0:
                    obitos_masc[idade] = np.random.binomial(int(pop_masc[idade]), self.prob_morte_masc[idade])

            # Calcula os óbitos
           # obitos_fem = np.random.binomial(pop_fem, self.prob_fem)
           # obitos_masc = np.random.binomial(pop_masc, self.prob_masc)

            # 3. Atualiza a população removendo os óbitos (operação vetorizada)
            pop_fem = pop_fem - obitos_fem
            pop_masc = pop_masc - obitos_masc

            # Garante que não haja valores negativos (operação vetorizada)
            pop_fem = np.maximum(0, pop_fem)
            pop_masc = np.maximum(0, pop_masc)

            # 4. Registra os óbitos deste ano
            total_obitos_ano = {
                'ano': ano, 
                'total': obitos_fem.sum() + obitos_masc.sum(),
                'feminino': obitos_fem.sum(), 
                'masculino': obitos_masc.sum()
            }
            obitos_anuais.append(total_obitos_ano)

            # 5. Registra a população ao final deste ano
            pop_ano = {
                'ano': ano, 
                'total': pop_fem.sum() + pop_masc.sum(),
                'feminino': pop_fem.sum(), 
                'masculino': pop_masc.sum()
            }
            populacao_anual.append(pop_ano)

            # 6. Envelhecimento: todos ficam um ano mais velhos
            if ano < anos:  # Não precisa envelhecer no último ano
                # Desloca a população para uma idade acima (exceto a última idade)
                pop_fem[1:] = pop_fem[:-1]
                pop_fem[0] = 0  # Zeramos a idade 0 (não há nascimentos)

                pop_masc[1:] = pop_masc[:-1]
                pop_masc[0] = 0  # Zeramos a idade 0 (não há nascimentos)

        return {
            'obitos': pd.DataFrame(obitos_anuais),
            'obitos_esperados': pd.DataFrame(obitos_esperados_anuais),
            'populacao': pd.DataFrame(populacao_anual)
        }

    def _executar_lote_simulacoes(self, params):
        """Função auxiliar para paralelização das simulações"""
        id_lote, n_simulacoes, anos, seed_base = params
        resultados_lote = []

        for i in range(n_simulacoes):
            # Usa uma seed diferente para cada simulação, mas reproduzível
            seed = seed_base + i
            resultado = self.simular_periodo(anos, seed)
            resultados_lote.append(resultado)

        return resultados_lote

    def executar_simulacoes(self, n_simulacoes=1000, anos=5, n_processos=None):
        """Executa múltiplas simulações em paralelo e compila os resultados"""
        # Define o número de processos se não especificado
        if n_processos is None:
            import multiprocessing
            n_processos = max(1, multiprocessing.cpu_count() - 1)

        # Define um número base para a seed para reprodutibilidade
        seed_base = 42

        # Divide as simulações em lotes para os processos
        simulacoes_por_processo = max(1, n_simulacoes // n_processos)
        lotes = []

        for i in range(n_processos):
            n_sim_lote = simulacoes_por_processo
            # Ajusta o último lote para garantir o número exato de simulações
            if i == n_processos - 1:
                n_sim_lote = n_simulacoes - i * simulacoes_por_processo

            if n_sim_lote > 0:
                lotes.append((i, n_sim_lote, anos, seed_base + i * simulacoes_por_processo))

        todos_resultados = []

        print(f"Executando {n_simulacoes} simulações em {len(lotes)} lotes usando {n_processos} processos...")

        # Executa as simulações em paralelo
        inicio = time.time()

        with ProcessPoolExecutor(max_workers=n_processos) as executor:
            futures = {executor.submit(self._executar_lote_simulacoes, params): i for i, params in enumerate(lotes)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processando lotes"):
                resultados_lote = future.result()
                todos_resultados.extend(resultados_lote)

        fim = time.time()
        print(f"Tempo de execução das simulações: {(fim - inicio):.2f} segundos")

        # Compila estatísticas de óbitos anuais (simulados)
        obitos_por_ano = {ano: {'total': [], 'feminino': [], 'masculino': []} for ano in range(1, anos + 1)}

        # Compila estatísticas de óbitos esperados anuais
        esperados_por_ano = {ano: {'total': [], 'feminino': [], 'masculino': []} for ano in range(1, anos + 1)}

        for resultado in todos_resultados:
            # Processa óbitos simulados
            for _, row in resultado['obitos'].iterrows():
                ano = row['ano']
                obitos_por_ano[ano]['total'].append(row['total'])
                obitos_por_ano[ano]['feminino'].append(row['feminino'])
                obitos_por_ano[ano]['masculino'].append(row['masculino'])

            # Processa óbitos esperados
            for _, row in resultado['obitos_esperados'].iterrows():
                ano = row['ano']
                esperados_por_ano[ano]['total'].append(row['total'])
                esperados_por_ano[ano]['feminino'].append(row['feminino'])
                esperados_por_ano[ano]['masculino'].append(row['masculino'])

        # Converte para DataFrame - estatísticas de óbitos simulados
        estatisticas_anuais = []
        for ano, dados in obitos_por_ano.items():
            estatisticas_anuais.append({
                'ano': ano,
                'media_total': np.mean(dados['total']),
                'desvio_total': np.std(dados['total']),
                'min_total': np.min(dados['total']),
                'max_total': np.max(dados['total']),
                'media_fem': np.mean(dados['feminino']),
                'desvio_fem': np.std(dados['feminino']),
                'media_masc': np.mean(dados['masculino']),
                'desvio_masc': np.std(dados['masculino'])
            })

        # Converte para DataFrame - estatísticas de óbitos esperados
        estatisticas_esperados = []
        for ano, dados in esperados_por_ano.items():
            estatisticas_esperados.append({
                'ano': ano,
                'media_total': np.mean(dados['total']),
                'desvio_total': np.std(dados['total']),
                'min_total': np.min(dados['total']),
                'max_total': np.max(dados['total']),
                'media_fem': np.mean(dados['feminino']),
                'desvio_fem': np.std(dados['feminino']),
                'media_masc': np.mean(dados['masculino']),
                'desvio_masc': np.std(dados['masculino'])
            })

        # Compila estatísticas de total acumulado de óbitos (5 anos)
        obitos_acumulados = []
        esperados_acumulados = []

        for sim_num, resultado in enumerate(todos_resultados):
            # Óbitos simulados acumulados
            df_sim = resultado['obitos']
            total_sim = {
                'simulacao': sim_num + 1,
                'total': df_sim['total'].sum(),
                'feminino': df_sim['feminino'].sum(),
                'masculino': df_sim['masculino'].sum()
            }
            obitos_acumulados.append(total_sim)

            # Óbitos esperados acumulados
            df_esp = resultado['obitos_esperados']
            total_esp = {
                'simulacao': sim_num + 1,
                'total': df_esp['total'].sum(),
                'feminino': df_esp['feminino'].sum(),
                'masculino': df_esp['masculino'].sum()
            }
            esperados_acumulados.append(total_esp)

        df_acumulados = pd.DataFrame(obitos_acumulados)
        df_esperados_acumulados = pd.DataFrame(esperados_acumulados)

        # Estatísticas dos óbitos simulados acumulados
        estatisticas_acumuladas = {
            'media_total': df_acumulados['total'].mean(),
            'desvio_total': df_acumulados['total'].std(),
            'min_total': df_acumulados['total'].min(),
            'max_total': df_acumulados['total'].max(),
            'media_fem': df_acumulados['feminino'].mean(),
            'desvio_fem': df_acumulados['feminino'].std(),
            'media_masc': df_acumulados['masculino'].mean(),
            'desvio_masc': df_acumulados['masculino'].std()
        }

        # Estatísticas dos óbitos esperados acumulados
        estatisticas_esperados_acumuladas = {
            'media_total': df_esperados_acumulados['total'].mean(),
            'desvio_total': df_esperados_acumulados['total'].std(),
            'min_total': df_esperados_acumulados['total'].min(),
            'max_total': df_esperados_acumulados['total'].max(),
            'media_fem': df_esperados_acumulados['feminino'].mean(),
            'desvio_fem': df_esperados_acumulados['feminino'].std(),
            'media_masc': df_esperados_acumulados['masculino'].mean(),
            'desvio_masc': df_esperados_acumulados['masculino'].std()
        }

        return {
            'resultados_brutos': todos_resultados,
            'estatisticas_anuais': pd.DataFrame(estatisticas_anuais),
            'estatisticas_esperados': pd.DataFrame(estatisticas_esperados),
            'acumulados': df_acumulados,
            'esperados_acumulados': df_esperados_acumulados,
            'estatisticas_acumuladas': estatisticas_acumuladas,
            'estatisticas_esperados_acumuladas': estatisticas_esperados_acumuladas
        }

    def gerar_relatorio(self, resultados, nome_arquivo=None):
        """Gera relatório e salva em Excel"""
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"relatorio_simulacao_{timestamp}.xlsx"

        # Cria um writer do Excel
        with pd.ExcelWriter(nome_arquivo) as writer:
            # Estatísticas anuais das simulações (óbitos simulados)
            resultados['estatisticas_anuais'].to_excel(writer, sheet_name='Estatisticas_Simulados', index=False)

            # Estatísticas anuais dos óbitos esperados
            resultados['estatisticas_esperados'].to_excel(writer, sheet_name='Estatisticas_Esperados', index=False)

            # Dados de óbitos esperados iniciais
            self.get_obitos_esperados_inicial_df().to_excel(writer, sheet_name='Obitos_Esperados_Inicial', index=False)

            # Amostra dos dados acumulados (primeiras 1000 simulações)
            amostra_sim = resultados['acumulados']#.head(1000)
            amostra_sim.to_excel(writer, sheet_name='Amostra_Simulacoes', index=False)

            # Amostra dos esperados acumulados
            amostra_esp = resultados['esperados_acumulados']#.head(1000)
            amostra_esp.to_excel(writer, sheet_name='Amostra_Esperados', index=False)

            # Resumo de estatísticas acumuladas
            df_resumo_sim = pd.DataFrame([resultados['estatisticas_acumuladas']])
            df_resumo_sim.to_excel(writer, sheet_name='Estatisticas_Acum_Simulados', index=False)

            # Resumo de estatísticas esperadas acumuladas
            df_resumo_esp = pd.DataFrame([resultados['estatisticas_esperados_acumuladas']])
            df_resumo_esp.to_excel(writer, sheet_name='Estatisticas_Acum_Esperados', index=False)

            # Dados da população original
            self.populacao_original.to_excel(writer, sheet_name='Populacao_Inicial', index=False)

            # Tábuas de mortalidade
            self.tabuas.to_excel(writer, sheet_name='Tabuas_Mortalidade', index=False)

            # Comparação entre óbitos esperados e simulados
            comparacao = pd.DataFrame({
                'ano': resultados['estatisticas_anuais']['ano'],
                'obitos_esperados_media': resultados['estatisticas_esperados']['media_total'],
                'obitos_simulados_media': resultados['estatisticas_anuais']['media_total'],
                'diferenca_absoluta': resultados['estatisticas_esperados']['media_total'] - resultados['estatisticas_anuais']['media_total'],
                'diferenca_percentual': (resultados['estatisticas_esperados']['media_total'] - resultados['estatisticas_anuais']['media_total']) / resultados['estatisticas_esperados']['media_total'] * 100
            })
            comparacao.to_excel(writer, sheet_name='Comparacao_Esperado_Simulado', index=False)

        # Imprime um relatório no console
        print("\n" + "="*50)
        print("RELATÓRIO DE SIMULAÇÃO DE MORTALIDADE")
        print("="*50)

        print("\nÓbitos esperados iniciais (primeiro ano):")
        print(self.get_obitos_esperados_inicial_df())

        print("\nEstatísticas de óbitos simulados por ano:")
        print(resultados['estatisticas_anuais'][['ano', 'media_total', 'desvio_total', 'min_total', 'max_total']])

        print("\nEstatísticas de óbitos esperados por ano (recalculados a cada ano):")
        print(resultados['estatisticas_esperados'][['ano', 'media_total', 'desvio_total', 'min_total', 'max_total']])

        print("\nComparação entre óbitos esperados e simulados:")
        print(comparacao)

        print("\nEstatísticas acumuladas de óbitos simulados (todos os anos):")
        for chave, valor in resultados['estatisticas_acumuladas'].items():
            print(f"{chave}: {valor:.2f}")

        print("\nEstatísticas acumuladas de óbitos esperados (todos os anos):")
        for chave, valor in resultados['estatisticas_esperados_acumuladas'].items():
            print(f"{chave}: {valor:.2f}")

        print("\nRelatório salvo em:", nome_arquivo)
        return nome_arquivo

    def visualizar_resultados(self, resultados):
        """Gera gráficos para visualização dos resultados"""
        # Configura o estilo
        plt.style.use('ggplot')

        # Figura 1: Óbitos médios por ano (simulado vs esperado)
        fig, ax = plt.subplots(figsize=(12, 7))

        anos = resultados['estatisticas_anuais']['ano']
        obitos_media = resultados['estatisticas_anuais']['media_total']
        obitos_desvio = resultados['estatisticas_anuais']['desvio_total']
        obitos_esperados_media = resultados['estatisticas_esperados']['media_total']
        obitos_esperados_desvio = resultados['estatisticas_esperados']['desvio_total']

        # Cria um gráfico de barras agrupadas
        bar_width = 0.35
        posicao_1 = np.arange(len(anos))
        posicao_2 = posicao_1 + bar_width

        # Barras para simulação
        ax.bar(posicao_1, obitos_media, width=bar_width, yerr=obitos_desvio, capsize=5,
               color='skyblue', edgecolor='blue', alpha=0.7, label='Óbitos Simulados (Média)')

        # Barras para valores esperados
        ax.bar(posicao_2, obitos_esperados_media, width=bar_width, yerr=obitos_esperados_desvio, capsize=5,
               color='lightgreen', edgecolor='green', alpha=0.7, label='Óbitos Esperados (Média)')

        ax.set_title('Comparação de Óbitos por Ano: Simulados vs Esperados', fontsize=14)
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel('Quantidade de Óbitos', fontsize=12)
        ax.set_xticks(posicao_1 + bar_width / 2)
        ax.set_xticklabels(anos)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adiciona valores sobre as barras
        for i, v in enumerate(obitos_media):
            ax.text(posicao_1[i], v + obitos_desvio[i] + 50, f'{v:.0f}',
                    ha='center', fontsize=9, fontweight='bold')

        for i, v in enumerate(obitos_esperados_media):
            ax.text(posicao_2[i], v + obitos_esperados_desvio[i] + 50, f'{v:.0f}',
                    ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig('comparacao_obitos_ano.png')

        # Figura 2: Histogramas comparativos de óbitos totais (simulados vs esperados)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histograma de óbitos simulados
        axes[0].hist(resultados['acumulados']['total'], bins=30,
                     color='skyblue', edgecolor='blue', alpha=0.7)

        media_sim = resultados['estatisticas_acumuladas']['media_total']
        desvio_sim = resultados['estatisticas_acumuladas']['desvio_total']

        axes[0].axvline(media_sim, color='red', linestyle='dashed', linewidth=2,
                        label=f'Média: {media_sim:.0f}')
        axes[0].axvline(media_sim + desvio_sim, color='orange', linestyle='dotted', linewidth=2,
                        label=f'Desvio: {desvio_sim:.0f}')
        axes[0].axvline(media_sim - desvio_sim, color='orange', linestyle='dotted', linewidth=2)

        axes[0].set_title('Distribuição do Total de Óbitos Simulados (5 Anos)', fontsize=14)
        axes[0].set_xlabel('Total de Óbitos', fontsize=12)
        axes[0].set_ylabel('Frequência', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Histograma de óbitos esperados
        axes[1].hist(resultados['esperados_acumulados']['total'], bins=30,
                     color='lightgreen', edgecolor='green', alpha=0.7)

        media_esp = resultados['estatisticas_esperados_acumuladas']['media_total']
        desvio_esp = resultados['estatisticas_esperados_acumuladas']['desvio_total']

        axes[1].axvline(media_esp, color='red', linestyle='dashed', linewidth=2,
                        label=f'Média: {media_esp:.0f}')
        axes[1].axvline(media_esp + desvio_esp, color='orange', linestyle='dotted', linewidth=2,
                        label=f'Desvio: {desvio_esp:.0f}')
        axes[1].axvline(media_esp - desvio_esp, color='orange', linestyle='dotted', linewidth=2)

        axes[1].set_title('Distribuição do Total de Óbitos Esperados (5 Anos)', fontsize=14)
        axes[1].set_xlabel('Total de Óbitos', fontsize=12)
        axes[1].set_ylabel('Frequência', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('distribuicao_obitos_total_comparativo.png')

        # Figura 3: Boxplot comparando óbitos por sexo (simulados e esperados)
        fig, ax = plt.subplots(figsize=(12, 7))

        data = [
            resultados['acumulados']['feminino'],
            resultados['esperados_acumulados']['feminino'],
            resultados['acumulados']['masculino'],
            resultados['esperados_acumulados']['masculino']
        ]

        labels = ['Simulados\nFeminino', 'Esperados\nFeminino', 'Simulados\nMasculino', 'Esperados\nMasculino']
        colors = ['lightblue', 'lightgreen', 'lightblue', 'lightgreen']

        boxprops = {'alpha': 0.7}
        box = ax.boxplot(data, labels=labels, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue', **boxprops),
                  whiskerprops=dict(color='blue'),
                  capprops=dict(color='blue'),
                  medianprops=dict(color='red'))

        # Altera as cores dos boxes
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title('Comparação de Óbitos por Sexo: Simulados vs Esperados (Total em 5 Anos)', fontsize=14)
        ax.set_ylabel('Total de Óbitos', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adiciona legenda personalizada
        import matplotlib.patches as mpatches
        sim_patch = mpatches.Patch(color='lightblue', label='Simulados')
        esp_patch = mpatches.Patch(color='lightgreen', label='Esperados')
        ax.legend(handles=[sim_patch, esp_patch])

        plt.tight_layout()
        plt.savefig('comparacao_obitos_por_sexo.png')

        # Figura 4: Evolução do desvio entre simulados e esperados por ano
        fig, ax = plt.subplots(figsize=(10, 6))

        anos = resultados['estatisticas_anuais']['ano']
        diferenca_percentual = (resultados['estatisticas_anuais']['media_total'] - resultados['estatisticas_esperados']['media_total']) / resultados['estatisticas_esperados']['media_total'] * 100

        ax.plot(anos, diferenca_percentual, 'o-', color='purple', linewidth=2, markersize=8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_title('Desvio Percentual entre Óbitos Simulados e Esperados por Ano', fontsize=14)
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel('Desvio Percentual (%)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adiciona rótulos aos pontos
        for i, value in enumerate(diferenca_percentual):
            ax.annotate(f'{value:.2f}%',
                       (anos[i], value),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=10,
                       fontweight='bold')

        plt.tight_layout()
        plt.savefig('desvio_entre_simulados_esperados.png')

        print("\nGráficos salvos como arquivos PNG.")

        # Retorna as figuras para exibição interativa, se necessário
        return plt.gcf()


def main():
    """Função principal para executar a simulação"""
    inicio_total = time.time()

    # Verifica se os arquivos existem
    arquivos_necessarios = ["populacao.xlsx", "tabuas.xlsx"]
    for arquivo in arquivos_necessarios:
        if not os.path.exists(arquivo):
            print(f"ERRO: O arquivo {arquivo} não foi encontrado.")
            print("Certifique-se de que os arquivos estejam no mesmo diretório que este script.")
            return

    # Parâmetros da simulação
    n_simulacoes = int(input("Número de simulações a executar (recomendado: 1000-10000): ") or 1000)
    anos = int(input("Número de anos em cada simulação (1 a 10)") or 5)
    # anos = 5  # Fixo conforme especificação

    # Determina o número de processos automaticamente
    import multiprocessing
    n_processos = max(1, multiprocessing.cpu_count() - 1)
    print(f"\nUtilizando {n_processos} processos para paralelização.")

    print(f"\nIniciando {n_simulacoes} simulações para períodos de {anos} anos...")

    # Instancia e executa a simulação
    simulador = SimulacaoMortalidade("populacao.xlsx", "tabuas.xlsx")

    # Exibe os óbitos esperados iniciais
    print("\nÓbitos esperados iniciais (primeiro ano):")
    print(simulador.get_obitos_esperados_inicial_df())

    # Executa as simulações
    resultados = simulador.executar_simulacoes(n_simulacoes, anos, n_processos)

    # Gera o relatório
    simulador.gerar_relatorio(resultados)

    # Gera visualizações
    simulador.visualizar_resultados(resultados)

    fim_total = time.time()
    print("\nSimulação concluída com sucesso!")
    print(f"Tempo total de execução: {(fim_total - inicio_total):.2f} segundos")


if __name__ == "__main__":
    main()
