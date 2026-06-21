import os
import itertools
from data_loader import carregar_dados_movielens, preparar_matrizes_treino_teste
from model_based import SVDRecommender
from evaluation import avaliar_modelo

def realizar_grid_search_svd():
    print("--- Iniciando Grid Search para o Regularized SVD ---")
    
    # 1. Carregar dados
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    df = carregar_dados_movielens(caminho_dados)
    matriz_treino, matriz_teste = preparar_matrizes_treino_teste(df, test_size=0.2)
    """
    2. Definir o Grid de hiperparâmetros
    Mantemos épocas fixas em 20.
    """
    grade = {
        'n_fatores': [100, 150],         # Matrizes gigantes
        'lr': [0.01],                    # Congelado no vencedor
        'reg': [0.005, 0.01, 0.015]      # Testando regularizações minúsculas
    }
    
    # itertools.product cria todas as combinações possíveis das listas acima
    combinacoes = list(itertools.product(grade['n_fatores'], grade['lr'], grade['reg']))
    total_modelos = len(combinacoes)
    
    melhor_rmse = float('inf')
    melhor_combinacao = None
    
    print(f"Total de combinações a serem testadas: {total_modelos}\n")
    
    # 3. Loop de treinamento
    for idx, (f, lr, r) in enumerate(combinacoes, 1):
        print("="*40)
        print(f"Modelo {idx}/{total_modelos} | Fatores: {f}, LR: {lr}, Reg: {r}")
        
        modelo = SVDRecommender(n_fatores=f, lr=lr, reg=r, n_epochs=20)
        modelo.fit(matriz_treino)
        
        rmse = avaliar_modelo(modelo, matriz_teste)
        print(f"-> RMSE no Teste para esta combinação: {rmse:.4f}")
        
        # Atualiza se encontrou um modelo melhor
        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_combinacao = (f, lr, r)
            print("   [!] NOVO MELHOR MODELO ENCONTRADO!")
            
    # 4. Resultado Final
    print("\n" + "#"*50)
    print("BUSCA CONCLUÍDA!")
    print(f"O menor RMSE alcançado pelo SVD foi: {melhor_rmse:.4f}")
    print(f"Hiperparâmetros Vencedores:")
    print(f"- Número de Fatores: {melhor_combinacao[0]}")
    print(f"- Learning Rate (lr): {melhor_combinacao[1]}")
    print(f"- Regularização (reg): {melhor_combinacao[2]}")
    print("#"*50)

if __name__ == "__main__":
    realizar_grid_search_svd()