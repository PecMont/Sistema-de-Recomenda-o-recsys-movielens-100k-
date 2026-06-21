import numpy as np
from sklearn.metrics import mean_squared_error
import math

def avaliar_modelo(modelo, matriz_teste):
    """
    Avalia o modelo percorrendo apenas os itens que estão no conjunto de teste.
    Calcula o RMSE (Root Mean Squared Error).
    """
    print("Iniciando avaliação... isso pode levar alguns segundos dependendo do tamanho do teste.")
    
    # Pega os índices apenas de onde existe avaliação no teste (> 0)
    usuarios_idx, itens_idx = matriz_teste.nonzero()
    
    notas_reais = []
    notas_preditas = []
    
    total_avaliacoes = len(usuarios_idx)
    contador = 0
    
    for u, i in zip(usuarios_idx, itens_idx):
        nota_real = matriz_teste[u, i]
        nota_predita = modelo.predict_nota(u, i)
        
        # Garante que a nota predita fique no limite de 1 a 5 
        nota_predita = max(1.0, min(5.0, nota_predita))
        
        notas_reais.append(nota_real)
        notas_preditas.append(nota_predita)
        
        contador += 1
        if contador % 5000 == 0:
            print(f"Progresso: {contador}/{total_avaliacoes} avaliações processadas...")
            
    # Calcula o RMSE 
    mse = mean_squared_error(notas_reais, notas_preditas)
    rmse = math.sqrt(mse)
    
    return rmse