# Filtragem Colaborativa: Sistemas de Recomendação (MovieLens 100k)

Este repositório contém o Trabalho Final da disciplina de Introdução à Teoria de Sistemas de Recomendação (DCC/UFRRJ). O objetivo do projeto é a implementação nativa, hiperparametrização e comparação de algoritmos clássicos de filtragem colaborativa.

## 📌 Sobre o Projeto

Conforme as especificações do trabalho, o projeto exigia a construção de algoritmos baseados em memória e em modelo, além da avaliação de seus desempenhos em prever notas de filmes utilizando a base de dados MovieLens 100k. 

Para garantir o máximo desempenho e compreensão matemática dos modelos, as arquiteturas foram construídas "do zero" utilizando rotinas vetorizadas com **NumPy**, mitigando a necessidade de laços nativos do Python sempre que possível.

### Algoritmos Implementados:
* **Filtro Baseado em Memória (User-Based):** Utiliza vizinhança entre usuários e Correlação de Pearson (similaridade de cosseno com matriz centralizada pela média) para neutralizar vieses de avaliação.
* **Filtro Baseado em Modelo (SVD Regularizado):** Fatoração de matrizes por Decomposição em Valores Singulares, otimizada através de Descida de Gradiente Estocástico (SGD) com regularização de Tikhonov.
* **Metodologia de Avaliação:** Particionamento Holdout (80/20) e métrica RMSE (Root Mean Squared Error).

---

## 🚀 Funcionalidades Extras Implementadas

Para enriquecer a análise e ir além das especificações básicas, os seguintes recursos adicionais foram desenvolvidos:

1. **Benchmarking de Mercado (Scikit-Surprise):** Implementação de rotinas de comparação direta entre os algoritmos nativos e a biblioteca de alto nível `scikit-surprise`. A análise confronta o trade-off entre a acurácia (RMSE) do ajuste fino nativo contra a eficiência computacional (Cython) da biblioteca.
2. **Visualizações Avançadas de Resultados:**
   Geração automática de gráficos de dispersão (com aplicação de *jitter* visual) para observar o fenômeno de regressão à média, além de Matrizes de Confusão para simular cenários reais de recomendação (binarizados em um limiar de nota $\ge$ 4.0).
3. **Análise de Sensibilidade e Hiperparametrização:**
   Rotinas de *Grid Search* exaustivo para isolar o ponto ótimo (Overfitting vs. Underfitting) de hiperparâmetros vitais, como o número ideal de Vizinhos (K), Fatores Latentes, Taxa de Aprendizado (LR) e Regularização.

---

## 🛠️ Como Instalar e Executar

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado. Recomenda-se o uso de um ambiente virtual (venv).

1. Clone este repositório:
```bash
git clone https://github.com/PecMont/Sistema-de-Recomenda-o-recsys-movielens-100k-.git
cd Sistema-de-Recomenda-o-recsys-movielens-100k-
```
2. nstale as dependências listadas no `requirements.txt`:
```bash
pip install -r requirements.txt
```
### Executando os Módulos
O projeto foi modularizado para facilitar a experimentação. A partir da raiz do projeto, você pode rodar os seguintes scripts:

#### 1. Comparação Principal: 
Roda o particionamento de dados, treina o User-Based e o SVD com os hiperparâmetros ideais e imprime o RMSE final de ambos.
```bash
python main.py
```
#### 2. Geração de Gráficos e Visualizações Avançadas:
Gera os gráficos de Dispersão, Matriz de Confusão e o gráfico de Análise de Sensibilidade (Fatores Latentes vs RMSE), salvando os resultados como arquivos PDF/PNG na raiz do projeto.
```bash
python src/visualizations.py
```
#### 3. Busca de Hiperparâmetros:
Para reproduzir os testes que definiram os melhores hiperparâmetros. (Nota: O Grid Search do SVD pode levar alguns minutos devido à natureza do processamento do SGD).
```bash
python src/tuning.py        # Analisa o impacto do número de Vizinhos (K)
python src/tuning_svd.py    # Busca extrema para o SVD (Fatores, LR, Reg)
```
#### 4. Benchmarking com Scikit-Surprise:
Treina os modelos equivalentes utilizando a biblioteca scikit-surprise para comparação de tempo de execução e acurácia. O script secundário gera as matrizes de confusão específicas desta biblioteca.
```bash
python src/compare_surprise.py
python src/visualizations_surprise.py
```
---
## 📂 Estrutura do Repositório

* `/data/ml-100k/` - Contém a base de dados do MovieLens (`u.data`).
* `/src/data_loader.py` - Scripts de importação e separação de matrizes (Treino/Teste).
* `/src/memory_based.py` - Implementação nativa do algoritmo User-Based.
* `/src/model_based.py` - Implementação nativa do algoritmo SVD.
* `/src/evaluation.py` - Módulo calculador de RMSE.
* `/src/tuning...` e `/src/visualizations...` - Scripts de análise sensível e geração de gráficos.
* `main.py` - Arquivo de execução central.
* `Artigo.pdf` - Relatório final formatado no template ACM com as conclusões metodológicas.

