# 🛡️ Credit Risk Model Simulator (PD Analytics)

Este é um simulador interativo desenvolvido em **Streamlit** para visualização e validação de métricas de classificação em modelos de **Risco de Crédito (Probability of Default)**.

O objetivo da aplicação é demonstrar como diferentes estratégias de *thresholding* impactam a eficiência do modelo na captura de inadimplentes (*Recall/TPR*) vs. o custo de oportunidade de negar bons clientes (*FPR*).

## 🚀 Funcionalidades

* **Simulação de Dados Realistas:** Geração de populações sintéticas com distribuições de PD concentradas próximas a zero, simulando o desbalanceamento real de carteiras de crédito.
* **Controle de Separabilidade:** Ajuste da qualidade do modelo (KS/AUC) para observar o comportamento das métricas em cenários de alta e baixa discriminação.
* **Comparativo de Estratégias:**
* **Threshold Fixo:** Aplicação de um corte estático de probabilidade.
* **Threshold por Percentil:** Seleção dos top $X\%$ piores scores (estratégia comum em comitês de crédito).


* **Visualização Avançada:** Gráfico de densidade com curvas de Bons e Maus (KDE) e sinalização das zonas de corte.
* **Métricas de Validação:** Tabela completa com KS, AUC, Acurácia, Precision, Recall (TPR), FPR e FNR.

## 📊 Conceitos Chave Explorados

A aplicação foca no *trade-off* entre:

* **TPR (True Positive Rate):** Sensibilidade do modelo em mitigar perdas.
* **FPR (False Positive Rate):** Métrica de custo de oportunidade (atrito com bons clientes).
* **FNR (False Negative Rate):** Representação do risco residual que entra no balanço.

## 🛠️ Instalação e Uso

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/credit-risk-simulator.git
cd credit-risk-simulator

```


2. **Instale as dependências:**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy

```


3. **Execute a aplicação:**
```bash
streamlit run app.py

```



## 📋 Requisitos

* Python 3.8+
* Streamlit
* Scikit-Learn
* Pandas / Numpy
* Matplotlib / Seaborn
* Scipy

## 🧠 Metodologia de Validação

Diferente de modelos de classificação genéricos, esta ferramenta evita o erro comum de utilizar o corte padrão de 0.5. Ela permite ao analista de risco identificar o **ponto de corte ótimo** com base na taxa de default da carteira e nos objetivos de negócio (expansão vs. conservadorismo).
