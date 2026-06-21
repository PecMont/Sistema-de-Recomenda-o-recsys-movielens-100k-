import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def gerar_visualizacoes_surprise(y_true, y_pred):
    """Gera o Scatter Plot e a Matriz de Confusão para as predições do Surprise"""
    print("Gerando gráficos de visualização (Surprise)...")
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Gráfico de Dispersão
    jitter_true = y_true + np.random.normal(0, 0.1, size=len(y_true))
    jitter_pred = y_pred + np.random.normal(0, 0.1, size=len(y_pred))
    
    axs[0].scatter(jitter_true, jitter_pred, alpha=0.1, color='green', s=2) 
    axs[0].plot([1, 5], [1, 5], 'r--', lw=2)
    axs[0].set_title('Notas Reais vs. Previstas (Surprise SVD)')
    axs[0].set_xlabel('Nota Real (com Jitter)')
    axs[0].set_ylabel('Nota Prevista (com Jitter)')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Matriz de Confusão 
    limiar = 4.0
    y_true_bin = (y_true >= limiar).astype(int)
    y_pred_bin = (y_pred >= limiar).astype(int)
    
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axs[1], 
                xticklabels=['< 4.0 (Negativo)', '>= 4.0 (Positivo)'],
                yticklabels=['< 4.0 (Negativo)', '>= 4.0 (Positivo)'])
    axs[1].set_title(f'Matriz de Confusão (Surprise | Limiar $\geq$ 4.0)')
    axs[1].set_ylabel('Classe Real')
    axs[1].set_xlabel('Classe Prevista')
    
    plt.tight_layout()
    # Salvando em PDF 
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grafico_resultados_surprise.pdf')
    plt.savefig(caminho, format='pdf', bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho}")
    plt.show()

if __name__ == "__main__":
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'data', 'ml-100k', 'u.data')
    
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(caminho_dados, reader=reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print("Treinando modelo Surprise SVD...")
    algo_svd = SVD(n_factors=150, n_epochs=20, lr_all=0.01, reg_all=0.015)
    algo_svd.fit(trainset)
    previsoes = algo_svd.test(testset)
    
    # O Surprise retorna uma lista de objetos. Tem que extrair a nota real (r_ui) e a estimada (est)
    y_true = np.array([pred.r_ui for pred in previsoes])
    y_pred = np.array([pred.est for pred in previsoes])
    
    gerar_visualizacoes_surprise(y_true, y_pred)