import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. PREPARAÇÃO DOS DADOS ---
dados = {
    'Aluno': ['Renan', 'Bruno', 'Baiano', 'Gordao', 'Vinicio', 'Gabriel', 'Comine', 'Gabigo', 'Toddy', 'Thiago'],
    'Python': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    'SQL':    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    'C#':     [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    'PHP':    [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}

# Transformando em DataFrame
df = pd.DataFrame(dados)
matriz_incidencia = df.set_index('Aluno')

print("=== 1. MATRIZ DE INCIDÊNCIA (Alunos x Habilidades) ===")
print(matriz_incidencia)
print("\n")

# Matriz de Similaridade (Aluno x Aluno)
matriz_similaridade = matriz_incidencia.dot(matriz_incidencia.T)
np.fill_diagonal(matriz_similaridade.values, 0)

# --- NOVO: PRINT DA MATRIZ DE SIMILARIDADE ---
print("=== 2. MATRIZ DE SIMILARIDADE (Aluno x Aluno) ===")
print(matriz_similaridade)
print("\n")


# Matriz de Coocorrência (Habilidade x Habilidade)
matriz_coocorrencia = matriz_incidencia.T.dot(matriz_incidencia)
np.fill_diagonal(matriz_coocorrencia.values, 0)

# --- NOVO: PRINT DA MATRIZ DE COOCORRÊNCIA ---
print("=== 3. MATRIZ DE COOCORRÊNCIA (Matéria x Matéria) ===")
print(matriz_coocorrencia)
print("\n")


# --- 2. FUNÇÕES AUXILIARES ---

def exibir_tabela_metricas(G, titulo):
    """Gera e imprime a tabela de métricas no terminal"""
    print(f"\n>>> TABELA DE MÉTRICAS: {titulo} <<<")
    
    # Cálculos das métricas
    degree = dict(nx.degree(G))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    # Criar DataFrame para exibição
    df_metricas = pd.DataFrame({
        'Grau': degree,
        'Intermediação': betweenness,
        'Proximidade': closeness
    })

    # Arredondar e Ordenar pelo Grau
    df_metricas = df_metricas.round(3)
    df_ordenado = df_metricas.sort_values(by='Grau', ascending=False)
    
    print(df_ordenado)
    print("-" * 60)

def analisar_grafo(adj_matrix, titulo_grafico, titulo_janela):
    """Cria o grafo, plota com nomes e chama a tabela de métricas."""
    
    # Define o título da janela do Windows
    plt.figure(figsize=(10, 8), num=titulo_janela)
    
    G = nx.from_pandas_adjacency(adj_matrix)
    
    pos = nx.spring_layout(G, seed=42) 
    
    # Desenha o grafo com os NOMES (with_labels=True)
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=2500, 
            font_size=10, 
            font_weight='bold', 
            edge_color='gray')
    
    # Desenha os pesos nas arestas
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    # Título interno do gráfico
    plt.title(titulo_grafico, fontsize=14, fontweight='bold')
    
    # Chama a função que mostra as tabelas no terminal
    exibir_tabela_metricas(G, titulo_grafico)
    
    plt.show()

# --- 3. EXECUÇÃO DOS GRAFOS PROJETADOS ---

# 1. Grafo de Similaridade
print("Gerando Grafo 1 (Similaridade)...")
analisar_grafo(matriz_similaridade, 
               "Grafo de Similaridade (Aluno x Aluno)", 
               "Grafo 1 - Rede de Alunos")

# 2. Grafo de Coocorrência
print("Gerando Grafo 2 (Coocorrência)...")
analisar_grafo(matriz_coocorrencia, 
               "Grafo de Coocorrência (Matéria x Matéria)", 
               "Grafo 2 - Rede de Habilidades")


# --- 4. GRAFO DE INCIDÊNCIA (BIPARTIDO) ---

print("Gerando Grafo 3 (Bipartido)...")
plt.figure(figsize=(12, 8), num="Grafo 3 - Bipartido")

B = nx.Graph()

alunos = list(matriz_incidencia.index)
habilidades = list(matriz_incidencia.columns)

# Adicionar nós (Alunos e Habilidades)
B.add_nodes_from(alunos, bipartite=0)
B.add_nodes_from(habilidades, bipartite=1)

# Adicionar arestas
for aluno in alunos:
    for habilidade in habilidades:
        if matriz_incidencia.loc[aluno, habilidade] == 1:
            B.add_edge(aluno, habilidade)

# Layout Bipartido
pos_bi = nx.bipartite_layout(B, alunos)

# Cores: Verde para Alunos, Laranja para Matérias
cores = ['lightgreen' if node in alunos else 'orange' for node in B.nodes]

nx.draw(B, pos_bi, 
        with_labels=True,        # Mostra os nomes
        node_color=cores, 
        node_size=2500, 
        font_size=10, 
        font_weight='bold', 
        edge_color='gray')

# Título atualizado
plt.title("Grafo Bipartido (Aluno x Matéria)", fontsize=14, fontweight='bold')
plt.show()