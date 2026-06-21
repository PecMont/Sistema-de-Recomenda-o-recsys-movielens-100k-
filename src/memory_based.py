import numpy as np

class FiltroColaborativoUserBased:
    def __init__(self, k_vizinhos=20):
        self.k_vizinhos = k_vizinhos
        self.matriz_treino = None
        self.medias_usuarios = None
        self.matriz_centrada = None
        self.matriz_similaridade = None

    def fit(self, matriz_treino):
        """
        Treina o modelo calculando a Correlação de Pearson 
        (Similaridade de Cosseno na matriz centralizada pela média).
        """
        self.matriz_treino = matriz_treino
        
        # 1. Calcular a média de cada usuário 
        mascara_avaliados = matriz_treino > 0
        somas_usuarios = np.sum(matriz_treino, axis=1)
        contagem_avaliacoes = np.sum(mascara_avaliados, axis=1)
        
        # Evitar divisão por zero 
        contagem_segura = np.where(contagem_avaliacoes == 0, 1, contagem_avaliacoes)
        self.medias_usuarios = somas_usuarios / contagem_segura
        
        # 2. Centralizar as notas subtraindo a média do usuário
        self.matriz_centrada = np.zeros_like(matriz_treino)
        # Onde a máscara for True, aplicamos a subtração
        for u in range(matriz_treino.shape[0]):
            itens_avaliados = mascara_avaliados[u]
            if np.any(itens_avaliados):
                self.matriz_centrada[u, itens_avaliados] = matriz_treino[u, itens_avaliados] - self.medias_usuarios[u]
                
        # 3. Calcular a Matriz de Similaridade com a matriz centrada
        epsilon = 1e-9
        normas = np.linalg.norm(self.matriz_centrada, axis=1, keepdims=True)
        
        # O produto escalar da matriz centrada resulta no numerador de Pearson
        dot_product = np.dot(self.matriz_centrada, self.matriz_centrada.T)
        self.matriz_similaridade = dot_product / (normas @ normas.T + epsilon)
        
        np.fill_diagonal(self.matriz_similaridade, 0)
        
        return self

    def predict_nota(self, user_idx, item_idx): # Preve a nota considerando o viés do usuário (média) e as notas centralizadas dos vizinhos.
     
        media_alvo = self.medias_usuarios[user_idx]
        
        # Se o usuário não tem avaliações no treino, retornamos o meio da escala
        if media_alvo == 0:
            return 3.0
            
        sim_usuario = self.matriz_similaridade[user_idx]
        usuarios_que_avaliaram = np.where(self.matriz_treino[:, item_idx] > 0)[0]
        
        # Se nenhum outro usuário avaliou este filme, retornamos a média do usuário alvo
        if len(usuarios_que_avaliaram) == 0:
            return media_alvo
            
        sim_filtrada = sim_usuario[usuarios_que_avaliaram]
        # Pegamos as notas JÁ CENTRALIZADAS dos vizinhos para este filme
        notas_centradas_filtradas = self.matriz_centrada[usuarios_que_avaliaram, item_idx]
        
        indices_top_k = np.argsort(sim_filtrada)[::-1][:self.k_vizinhos]
        top_sim = sim_filtrada[indices_top_k]
        top_notas_centradas = notas_centradas_filtradas[indices_top_k]
        
        soma_sim = np.sum(np.abs(top_sim))
        
        # Se a similaridade for zero, predizemos apenas a média do usuário alvo
        if soma_sim == 0:
            return media_alvo
            
        # 4. Cálculo final: Média do Alvo + Média Ponderada das Notas Centralizadas dos Vizinhos
        predicao_centrada = np.sum(top_notas_centradas * top_sim) / soma_sim
        predicao_final = media_alvo + predicao_centrada
        
        return predicao_final