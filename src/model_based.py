import numpy as np

class SVDRecommender:
    def __init__(self, n_fatores=20, lr=0.01, reg=0.1, n_epochs=20):
        """
        n_fatores: Número de fatores latentes.
        lr: Learning rate.
        reg: Termo de regularização para evitar overfitting.
        n_epochs: Quantas vezes o algoritmo vai passar por toda a base de treino.
        """
        self.n_fatores = n_fatores
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.P = None
        self.Q = None

    def fit(self, matriz_treino):
        n_users, n_items = matriz_treino.shape
        
        # Inicializando P e Q com valores aleatórios pequenos 
        self.P = np.random.normal(scale=1./self.n_fatores, size=(n_users, self.n_fatores))
        self.Q = np.random.normal(scale=1./self.n_fatores, size=(n_items, self.n_fatores))
        
        usuarios_idx, itens_idx = matriz_treino.nonzero()
        
        print(f"Iniciando treinamento SVD ({self.n_epochs} épocas)...")
        
        for epoch in range(self.n_epochs):
            erro_quadratico_total = 0
            
            for u, i in zip(usuarios_idx, itens_idx):
                r_ui = matriz_treino[u, i]
                
                # Produto escalar
                predicao = np.dot(self.P[u, :], self.Q[i, :])
                
                # Calculando o erro
                e_ui = r_ui - predicao
                erro_quadratico_total += e_ui**2
                
                # Backup dos valores antigos para atualizar simultaneamente
                P_u_old = self.P[u, :]
                Q_i_old = self.Q[i, :]
                
                # Regra de atualização do Gradiente Descendente
                self.P[u, :] = P_u_old + self.lr * (e_ui * Q_i_old - self.reg * P_u_old)
                self.Q[i, :] = Q_i_old + self.lr * (e_ui * P_u_old - self.reg * Q_i_old)
                
            rmse_treino = np.sqrt(erro_quadratico_total / len(usuarios_idx))
            print(f"Época {epoch+1}/{self.n_epochs} | RMSE Treino: {rmse_treino:.4f}")
            
        return self

    def predict_nota(self, user_idx, item_idx):
        """
        A predição é a multiplicação dos vetores latentes do usuário e do item.
        Se por algum motivo o índice do item estiver fora, garantimos que não quebre e retorne uma nota mediana.
        """
        try:
            predicao = np.dot(self.P[user_idx, :], self.Q[item_idx, :])
            return predicao
        except IndexError:
            return 3.0