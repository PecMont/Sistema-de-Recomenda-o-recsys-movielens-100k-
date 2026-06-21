import os
from src.data_loader import carregar_dados_movielens, preparar_matrizes_treino_teste
from src.memory_based import FiltroColaborativoUserBased
from src.model_based import SVDRecommender
from src.evaluation import avaliar_modelo

def main():
    print("--- Comparação de Algoritmos de Recomendação ---")
    
    # 1. Carregando os Dados
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, 'data', 'ml-100k', 'u.data')
    df = carregar_dados_movielens(caminho_dados)
    matriz_treino, matriz_teste = preparar_matrizes_treino_teste(df, test_size=0.2)
    
    print("\n" + "="*50)
    print("1. TESTANDO BASEADO EM MEMÓRIA (USER-BASED COM PEARSON)")
    print("="*50)
    modelo_memoria = FiltroColaborativoUserBased(k_vizinhos=30)
    modelo_memoria.fit(matriz_treino)
    rmse_memoria = avaliar_modelo(modelo_memoria, matriz_teste)
    print(f"-> RMSE Final Baseado em Memória: {rmse_memoria:.4f}")
    
    print("\n" + "="*50)
    print("2. TESTANDO BASEADO EM MODELO (REGULARIZED SVD)")
    print("="*50)
    # Inicializando com hiperparâmetros padrão do SVD
    modelo_svd = SVDRecommender(n_fatores=20, lr=0.01, reg=0.1, n_epochs=20)
    modelo_svd.fit(matriz_treino)
    print("\nAvaliando predições do SVD na base de teste...")
    rmse_svd = avaliar_modelo(modelo_svd, matriz_teste)
    print(f"-> RMSE Final Baseado em Modelo (SVD): {rmse_svd:.4f}")
    
    print("\n" + "*"*50)
    print("RESUMO DA COMPARAÇÃO:")
    print(f"User-Based CF: {rmse_memoria:.4f}")
    print(f"SVD (Matrix Factorization): {rmse_svd:.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()