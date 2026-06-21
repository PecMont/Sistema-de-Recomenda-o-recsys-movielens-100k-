import os
import time
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

def comparar_com_surprise():
    print("--- COMPARAÇÃO COM BIBLIOTECA DE ALTO NÍVEL (SURPRISE) ---")
    
    # 1. Carregando os dados 
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(caminho_dados, reader=reader)
    
    # Separando 80/20 
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("1. SURPRISE: USER-BASED CF (Pearson, K=30)")
    print("="*50)
    # Configurando para usar Pearson 
    sim_options = {'name': 'pearson', 'user_based': True}
    algo_knn = KNNBasic(k=30, sim_options=sim_options)
    
    inicio_knn = time.time()
    algo_knn.fit(trainset)
    previsoes_knn = algo_knn.test(testset)
    tempo_knn = time.time() - inicio_knn
    
    rmse_knn = accuracy.rmse(previsoes_knn, verbose=False)
    print(f"-> RMSE Surprise User-Based: {rmse_knn:.4f}")
    print(f"-> Tempo de execução (Treino + Teste): {tempo_knn:.2f} segundos")
    
    print("\n" + "="*50)
    print("2. SURPRISE: SVD (Fatores=150, lr=0.01, reg=0.015)")
    print("="*50)
    # Colocando exatamente os mesmos hiperparâmetros que encontramos na nossa Busca 
    algo_svd = SVD(n_factors=150, n_epochs=20, lr_all=0.01, reg_all=0.015)
    
    inicio_svd = time.time()
    algo_svd.fit(trainset)
    previsoes_svd = algo_svd.test(testset)
    tempo_svd = time.time() - inicio_svd
    
    rmse_svd = accuracy.rmse(previsoes_svd, verbose=False)
    print(f"-> RMSE Surprise SVD: {rmse_svd:.4f}")
    print(f"-> Tempo de execução (Treino + Teste): {tempo_svd:.2f} segundos")

if __name__ == "__main__":
    comparar_com_surprise()