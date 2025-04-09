"""Script para realizar simulações de mortalidade em uma população, conforme uma tábua biométrica.
Gerado pelo Claude IA - versão 0.1"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimulacaoMortalidade:
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

    def simular_periodo(self, anos=5):
        """Realiza uma simulação de mortalidade para um período de anos"""
        # Copia a população original para não alterá-la
        populacao = self.populacao_original.copy()

        # Inicializa o contador de óbitos por ano
        obitos_anuais = []
        populacao_anual = []

        # Registra população inicial
        pop_inicial = {'ano': 0,
                       'total': populacao['feminino'].sum() + populacao['masculino'].sum(),
                       'feminino': populacao['feminino'].sum(), 
                       'masculino': populacao['masculino'].sum()}
        populacao_anual.append(pop_inicial)

        for ano in range(1, anos + 1):
            # Calcula os óbitos para cada idade e sexo
            obitos_fem = np.zeros(len(populacao))
            obitos_masc = np.zeros(len(populacao))

            for i, row in populacao.iterrows():
                idade = row['idade']
                fem = row['feminino']
                masc = row['masculino']

                # Encontra as probabilidades de morte para esta idade
                prob_morte = self.tabuas[self.tabuas['idade'] == idade]
                if not prob_morte.empty:
                    prob_fem = prob_morte['prob_morte_fem'].values[0]
                    prob_masc = prob_morte['prob_morte_masc'].values[0]

                    # Simula os óbitos usando distribuição binomial
                    if fem > 0:
                        obitos_fem[i] = np.random.binomial(int(fem), prob_fem)
                    if masc > 0:
                        obitos_masc[i] = np.random.binomial(int(masc), prob_masc)

            # Atualiza a população removendo os óbitos
            populacao['feminino'] = populacao['feminino'] - obitos_fem
            populacao['masculino'] = populacao['masculino'] - obitos_masc

            # Garante que não haja valores negativos
            populacao['feminino'] = np.maximum(0, populacao['feminino'])
            populacao['masculino'] = np.maximum(0, populacao['masculino'])

            # Registra os óbitos deste ano
            total_obitos_ano = {'ano': ano,
                                'total': sum(obitos_fem) + sum(obitos_masc),
                                'feminino': sum(obitos_fem), 
                                'masculino': sum(obitos_masc)}
            obitos_anuais.append(total_obitos_ano)

            # Registra a população ao final deste ano
            pop_ano = {'ano': ano,
                       'total': populacao['feminino'].sum() + populacao['masculino'].sum(),
                       'feminino': populacao['feminino'].sum(), 
                       'masculino': populacao['masculino'].sum()}
            populacao_anual.append(pop_ano)

            # Envelhecimento: todos ficam um ano mais velhos
            if ano < anos:  # Não precisa envelhecer no último ano
                # Cria nova estrutura populacional para o próximo ano
                nova_populacao = pd.DataFrame({
                    'idade': list(range(self.idade_maxima + 1)),
                    'feminino': np.zeros(self.idade_maxima + 1),
                    'masculino': np.zeros(self.idade_maxima + 1)
                })

                # Transfere a população para uma idade acima
                for i, row in populacao.iterrows():
                    idade_atual = row['idade']
                    if idade_atual < self.idade_maxima:
                        nova_populacao.loc[nova_populacao['idade'] == idade_atual + 1, 'feminino'] += row['feminino']
                        nova_populacao.loc[nova_populacao['idade'] == idade_atual + 1, 'masculino'] += row['masculino']

                populacao = nova_populacao

        return {'obitos': pd.DataFrame(obitos_anuais), 'populacao': pd.DataFrame(populacao_anual)}

    def executar_simulacoes(self, n_simulacoes=1000, anos=5):
        """Executa múltiplas simulações e compila os resultados"""
        todos_resultados = []

        # Barra de progresso para acompanhar as simulações
        for _ in tqdm(range(n_simulacoes), desc="Executando simulações"):
            resultado = self.simular_periodo(anos)
            todos_resultados.append(resultado)

        # Compila estatísticas de óbitos anuais
        # Formato: {'ano': [resultados de todas simulações], ...}
        obitos_compilados = {
            'ano': [],
            'total': [],
            'feminino': [],
            'masculino': []
        }

        for sim_num in range(n_simulacoes):
            for _, row in todos_resultados[sim_num]['obitos'].iterrows():
                obitos_compilados['ano'].append(row['ano'])
                obitos_compilados['total'].append(row['total'])
                obitos_compilados['feminino'].append(row['feminino'])
                obitos_compilados['masculino'].append(row['masculino'])

        df_obitos = pd.DataFrame(obitos_compilados)

        # Agrupa por ano para estatísticas
        estatisticas_anuais = df_obitos.groupby('ano').agg(
            media_total=('total', 'mean'),
            desvio_total=('total', 'std'),
            min_total=('total', 'min'),
            max_total=('total', 'max'),
            media_fem=('feminino', 'mean'),
            desvio_fem=('feminino', 'std'),
            media_masc=('masculino', 'mean'),
            desvio_masc=('masculino', 'std')
        ).reset_index()

        # Compila estatísticas de total acumulado de óbitos (5 anos)
        obitos_acumulados = []
        for sim_num in range(n_simulacoes):
            df_sim = todos_resultados[sim_num]['obitos']
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
            'estatisticas_anuais': estatisticas_anuais,
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
            amostra = resultados['acumulados'].head(1000)
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

    print(f"\nIniciando {n_simulacoes} simulações para períodos de {anos} anos...")

    # Instancia e executa a simulação
    simulador = SimulacaoMortalidade("populacao.xlsx", "tabuas.xlsx")
    resultados = simulador.executar_simulacoes(n_simulacoes, anos)

    # Gera o relatório
    simulador.gerar_relatorio(resultados)

    # Gera visualizações
    simulador.visualizar_resultados(resultados)

    print("\nSimulação concluída com sucesso!")


if __name__ == "__main__":
    main()
