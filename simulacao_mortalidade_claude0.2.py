"""Script para realizar simulações de mortalidade em uma população, conforme uma tábua biométrica.
Gerado pelo Claude IA - versão 0.2"""

import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimulacaoMortalidade:
    """Módulo que faz a simulação"""
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
        self.pop_fem_array = np.array(self.populacao_original['feminino'])
        self.pop_masc_array = np.array(self.populacao_original['masculino'])

        # Cria dicionários de probabilidades por idade para acesso rápido
        self.prob_morte_fem = {}
        self.prob_morte_masc = {}

        for _, row in self.tabuas.iterrows():
            idade = int(row['idade'])
            self.prob_morte_fem[idade] = row['prob_morte_fem']
            self.prob_morte_masc[idade] = row['prob_morte_masc']

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

            # Atualiza a população removendo os óbitos (operação vetorizada)
            pop_fem = pop_fem - obitos_fem
            pop_masc = pop_masc - obitos_masc

            # Garante que não haja valores negativos (operação vetorizada)
            pop_fem = np.maximum(0, pop_fem)
            pop_masc = np.maximum(0, pop_masc)

            # Registra os óbitos deste ano
            total_obitos_ano = {
                'ano': ano, 
                'total': obitos_fem.sum() + obitos_masc.sum(),
                'feminino': obitos_fem.sum(), 
                'masculino': obitos_masc.sum()
            }
            obitos_anuais.append(total_obitos_ano)

            # Registra a população ao final deste ano
            pop_ano = {
                'ano': ano, 
                'total': pop_fem.sum() + pop_masc.sum(),
                'feminino': pop_fem.sum(), 
                'masculino': pop_masc.sum()
            }
            populacao_anual.append(pop_ano)

            # Envelhecimento: todos ficam um ano mais velhos
            if ano < anos:  # Não precisa envelhecer no último ano
                # Desloca a população para uma idade acima (exceto a última idade)
                # Esta é uma operação otimizada em vez de criar um novo DataFrame
                pop_fem[1:] = pop_fem[:-1]
                pop_fem[0] = 0  # Zeramos a idade 0 (não há nascimentos)

                pop_masc[1:] = pop_masc[:-1]
                pop_masc[0] = 0  # Zeramos a idade 0 (não há nascimentos)

        return {
            'obitos': pd.DataFrame(obitos_anuais), 
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

        # Compila estatísticas de óbitos anuais
        # Usa uma abordagem otimizada para compilar os resultados
        obitos_por_ano = {ano: {'total': [], 'feminino': [], 'masculino': []} for ano in range(1, anos + 1)}

        for resultado in todos_resultados:
            for _, row in resultado['obitos'].iterrows():
                ano = row['ano']
                obitos_por_ano[ano]['total'].append(row['total'])
                obitos_por_ano[ano]['feminino'].append(row['feminino'])
                obitos_por_ano[ano]['masculino'].append(row['masculino'])

        # Converte para DataFrame
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

        # Compila estatísticas de total acumulado de óbitos (5 anos)
        obitos_acumulados = []
        for sim_num, resultado in enumerate(todos_resultados):
            df_sim = resultado['obitos']
            total_sim = {
                'simulacao': sim_num + 1,
                'total': df_sim['total'].sum(),
                'feminino': df_sim['feminino'].sum(),
                'masculino': df_sim['masculino'].sum()
            }
            obitos_acumulados.append(total_sim)

        df_acumulados = pd.DataFrame(obitos_acumulados)

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

        return {
            'resultados_brutos': todos_resultados,
            'estatisticas_anuais': pd.DataFrame(estatisticas_anuais),
            'acumulados': df_acumulados,
            'estatisticas_acumuladas': estatisticas_acumuladas
        }

    def gerar_relatorio(self, resultados, nome_arquivo=None):
        """Gera relatório e salva em Excel"""
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"relatorio_simulacao_{timestamp}.xlsx"

        # Cria um writer do Excel
        with pd.ExcelWriter(nome_arquivo) as writer:
            # Estatísticas anuais
            resultados['estatisticas_anuais'].to_excel(writer, sheet_name='Estatisticas_Anuais', index=False)

            # Amostra dos dados acumulados (primeiras 1000 simulações)
            amostra = resultados['acumulados'].head(100000)
            amostra.to_excel(writer, sheet_name='Amostra_Simulações', index=False)

            # Resumo de estatísticas acumuladas
            df_resumo = pd.DataFrame([resultados['estatisticas_acumuladas']])
            df_resumo.to_excel(writer, sheet_name='Estatisticas_Acumuladas', index=False)

            # Dados da população original
            self.populacao_original.to_excel(writer, sheet_name='Populacao_Inicial', index=False)

            # Tábuas de mortalidade
            self.tabuas.to_excel(writer, sheet_name='Tabuas_Mortalidade', index=False)

        # Imprime também um relatório no console
        print("\n" + "="*50)
        print("RELATÓRIO DE SIMULAÇÃO DE MORTALIDADE")
        print("="*50)

        print("\nEstatísticas de óbitos por ano:")
        print(resultados['estatisticas_anuais'][['ano', 'media_total', 'desvio_total', 'min_total', 'max_total']])

        print("\nEstatísticas acumuladas (todos os anos):")
        for chave, valor in resultados['estatisticas_acumuladas'].items():
            print(f"{chave}: {valor:.2f}")

        print("\nRelatório salvo em:", nome_arquivo)
        return nome_arquivo

    def visualizar_resultados(self, resultados):
        """Gera gráficos para visualização dos resultados"""
        # Configura o estilo
        plt.style.use('ggplot')

        # Figura 1: Óbitos médios por ano
        fig, ax = plt.subplots(figsize=(10, 6))

        anos = resultados['estatisticas_anuais']['ano']
        obitos_media = resultados['estatisticas_anuais']['media_total']
        obitos_desvio = resultados['estatisticas_anuais']['desvio_total']

        ax.bar(anos, obitos_media, yerr=obitos_desvio, capsize=5,
               color='skyblue', edgecolor='blue', alpha=0.7)

        ax.set_title('Média de Óbitos por Ano (com desvio padrão)', fontsize=14)
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel('Quantidade de Óbitos', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adiciona valores sobre as barras
        for i, v in enumerate(obitos_media):
            ax.text(anos[i], v + obitos_desvio[i] + 100, f'{v:.0f}',
                    ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('obitos_por_ano.png')

        # Figura 2: Histograma de óbitos totais
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(resultados['acumulados']['total'], bins=30,
                color='lightgreen', edgecolor='darkgreen', alpha=0.7)

        media = resultados['estatisticas_acumuladas']['media_total']
        desvio = resultados['estatisticas_acumuladas']['desvio_total']

        ax.axvline(media, color='red', linestyle='dashed', linewidth=2,
                  label=f'Média: {media:.0f}')
        ax.axvline(media + desvio, color='orange', linestyle='dotted', linewidth=2,
                  label=f'Desvio: {desvio:.0f}')
        ax.axvline(media - desvio, color='orange', linestyle='dotted', linewidth=2)

        ax.set_title('Distribuição do Total de Óbitos em 5 Anos', fontsize=14)
        ax.set_xlabel('Total de Óbitos', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('distribuicao_obitos_total.png')

        # Figura 3: Boxplot comparando óbitos por sexo
        fig, ax = plt.subplots(figsize=(10, 6))

        data = [resultados['acumulados']['feminino'], resultados['acumulados']['masculino']]
        labels = ['Feminino', 'Masculino']

        ax.boxplot(data, labels=labels, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='blue'),
                  capprops=dict(color='blue'),
                  medianprops=dict(color='red'))

        ax.set_title('Comparação de Óbitos por Sexo (Total em 5 Anos)', fontsize=14)
        ax.set_ylabel('Total de Óbitos', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('comparacao_obitos_por_sexo.png')

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
    anos = 5  # Fixo conforme especificação

    # Determina o número de processos automaticamente
    import multiprocessing
    n_processos = max(1, multiprocessing.cpu_count() - 1)
    print(f"\nUtilizando {n_processos} processos para paralelização.")

    print(f"\nIniciando {n_simulacoes} simulações para períodos de {anos} anos...")

    # Instancia e executa a simulação
    simulador = SimulacaoMortalidade("populacao.xlsx", "tabuas.xlsx")
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
