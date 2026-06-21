import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import carregar_dados_movielens, preparar_matrizes_treino_teste
from model_based import SVDRecommender

def obter_previsoes_reais(modelo, matriz_teste):
    """Função auxiliar para extrair arrays de y_true e y_pred"""
    usuarios_idx, itens_idx = matriz_teste.nonzero()
    y_true = []
    y_pred = []
    
    for u, i in zip(usuarios_idx, itens_idx):
        y_true.append(matriz_teste[u, i])
        pred = modelo.predict_nota(u, i)
        y_pred.append(max(1.0, min(5.0, pred))) # Limita entre 1 e 5
        
    return np.array(y_true), np.array(y_pred)

def gerar_visualizacoes_avancadas(y_true, y_pred):
    """Gera o Scatter Plot e a Matriz de Confusão"""
    print("Gerando gráficos de visualização avançada...")
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Gráfico de Dispersão (Scatter Plot)
    jitter_true = y_true + np.random.normal(0, 0.1, size=len(y_true))
    jitter_pred = y_pred + np.random.normal(0, 0.1, size=len(y_pred))
    
    axs[0].scatter(jitter_true, jitter_pred, alpha=0.1, color='blue', s=2)
    axs[0].plot([1, 5], [1, 5], 'r--', lw=2) # Linha de perfeição
    axs[0].set_title('Notas Reais vs. Previstas (SVD)')
    axs[0].set_xlabel('Nota Real (com Jitter)')
    axs[0].set_ylabel('Nota Prevista (com Jitter)')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Matriz de Confusão 
    limiar = 4.0
    y_true_bin = (y_true >= limiar).astype(int)
    y_pred_bin = (y_pred >= limiar).astype(int)
    
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1], 
                xticklabels=['< 4.0 (Negativo)', '>= 4.0 (Positivo)'],
                yticklabels=['< 4.0 (Negativo)', '>= 4.0 (Positivo)'])
    axs[1].set_title(f'Matriz de Confusão (Limiar $\geq$ 4.0)')
    axs[1].set_ylabel('Classe Real')
    axs[1].set_xlabel('Classe Prevista')
    
    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grafico_resultados.png')
    plt.savefig(caminho.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho}")
    plt.show()

def analise_sensibilidade_fatores(matriz_treino, matriz_teste):
    """Gera um gráfico mostrando o impacto de diferentes fatores latentes"""
    print("\nIniciando Análise de Sensibilidade (Fatores Latentes)...")
    
    fatores_lista = [10, 20, 40, 80, 150]
    erros_rmse = []
    
    for f in fatores_lista:
        print(f"Treinando SVD com {f} fatores...")
        # Usando a melhor lr e reg que você encontrou na busca 
        modelo = SVDRecommender(n_fatores=f, lr=0.01, reg=0.015, n_epochs=20)
        modelo.fit(matriz_treino)
        
        y_true, y_pred = obter_previsoes_reais(modelo, matriz_teste)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        erros_rmse.append(rmse)
        print(f"RMSE para {f} fatores: {rmse:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(fatores_lista, erros_rmse, marker='s', linestyle='-', color='purple', linewidth=2)
    plt.title('Análise de Sensibilidade: Fatores Latentes vs RMSE')
    plt.xlabel('Número de Fatores Latentes')
    plt.ylabel('RMSE (Menor é Melhor)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grafico_sensibilidade_svd.png')
    plt.savefig(caminho)
    print(f"Gráfico salvo em: {caminho}")
    plt.show()

if __name__ == "__main__":
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    df = carregar_dados_movielens(caminho_dados)
    matriz_treino, matriz_teste = preparar_matrizes_treino_teste(df, test_size=0.2)
    
    # 1. Gerar Visualizações Avançadas com o melhor modelo
    print("--- 1. PREPARANDO VISUALIZAÇÕES AVANÇADAS ---")
    melhor_modelo_svd = SVDRecommender(n_fatores=150, lr=0.01, reg=0.015, n_epochs=20)
    melhor_modelo_svd.fit(matriz_treino)
    y_true, y_pred = obter_previsoes_reais(melhor_modelo_svd, matriz_teste)
    gerar_visualizacoes_avancadas(y_true, y_pred)
    
    # 2. Gerar Análise de Sensibilidade
    print("\n--- 2. PREPARANDO ANÁLISE DE SENSIBILIDADE ---")
    analise_sensibilidade_fatores(matriz_treino, matriz_teste)