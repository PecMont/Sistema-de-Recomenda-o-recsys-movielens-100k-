import os
import matplotlib.pyplot as plt
from data_loader import carregar_dados_movielens, preparar_matrizes_treino_teste
from memory_based import FiltroColaborativoUserBased
from evaluation import avaliar_modelo

def realizar_grid_search_k():
    print("--- Iniciando Hiperparametrização do User-Based ---")
    
    # 1. Carregar dados
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    df = carregar_dados_movielens(caminho_dados)
    matriz_treino, matriz_teste = preparar_matrizes_treino_teste(df, test_size=0.2)
    
    # 2. Definir os valores de K que queremos testar
    valores_k = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]
    resultados_rmse = []
    
    # 3. Loop de treinamento e avaliação
    for k in valores_k:
        print(f"\nTestando K = {k}...")
        modelo = FiltroColaborativoUserBased(k_vizinhos=k)
        modelo.fit(matriz_treino)
        
        rmse = avaliar_modelo(modelo, matriz_teste)
        resultados_rmse.append(rmse)
        print(f"RMSE para K={k} finalizado: {rmse:.4f}")
        
    # 4. Descobrir qual foi o melhor K
    melhor_rmse = min(resultados_rmse)
    melhor_k = valores_k[resultados_rmse.index(melhor_rmse)]
    
    print("\n" + "="*40)
    print(f"MELHOR RESULTADO: K={melhor_k} com RMSE de {melhor_rmse:.4f}")
    print("="*40)
    
    # 5. Gerar Gráfico para o Relatório do Professor
    plt.figure(figsize=(10, 6))
    plt.plot(valores_k, resultados_rmse, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Impacto do Número de Vizinhos (K) no Erro de Previsão', fontsize=14)
    plt.xlabel('Número de Vizinhos (K)', fontsize=12)
    plt.ylabel('RMSE (Menor é Melhor)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Destacar o melhor ponto no gráfico
    plt.plot(melhor_k, melhor_rmse, marker='o', color='red', markersize=8, label=f'Melhor K ({melhor_k})')
    plt.legend()
    
    # Salvar o gráfico na pasta raiz para você usar no relatório
    caminho_grafico = os.path.join(diretorio_atual, '..', 'grafico_tuning_k.png')
    plt.savefig(caminho_grafico.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"\nGráfico salvo com sucesso em: {caminho_grafico}")
    plt.show()

if __name__ == "__main__":
    realizar_grid_search_k()