import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def carregar_dados_movielens(caminho_arquivo):  #Carrega o arquivo u.data e retorna o DataFrame.

    # Definindo o nome das colunas
    nomes_colunas = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Lendo o arquivo 
    df = pd.read_csv(caminho_arquivo, sep='\t', names=nomes_colunas)
    
    # O timestamp não é necessário para os modelos básicos de matriz
    df = df.drop('timestamp', axis=1)
    
    return df

def preparar_matrizes_treino_teste(df, test_size=0.2, random_state=42): # Divide os dados em treino e teste e cria as matrizes de Usuário x Item.

    # Dividindo as interações (garantindo que não vamos treinar com dados de teste)
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Determinando o número de usuários e itens
    n_users = df['user_id'].max()
    n_items = df['item_id'].max()
    
    # Inicializando as matrizes com zeros
    matriz_treino = np.zeros((n_users, n_items))
    matriz_teste = np.zeros((n_users, n_items))
    
    # Preenchendo a matriz de treino 
    for row in treino_df.itertuples():
        matriz_treino[row.user_id - 1, row.item_id - 1] = row.rating
        
    # Preenchendo a matriz de teste
    for row in teste_df.itertuples():
        matriz_teste[row.user_id - 1, row.item_id - 1] = row.rating
        
    return matriz_treino, matriz_teste

# Teste
if __name__ == "__main__":
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    caminho = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    
    df = carregar_dados_movielens(caminho)
    print(f"Total de avaliações carregadas: {len(df)}")
    print(df.head())
    
    matriz_treino, matriz_teste = preparar_matrizes_treino_teste(df)
    print(f"\nFormato da Matriz de Treino: {matriz_treino.shape}")