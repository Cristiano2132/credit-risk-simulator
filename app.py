import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ks_2samp

# 1. Configuração da Página e Estilo
st.set_page_config(page_title="Credit Risk Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background-color: #f1f3f6;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e0e4e8;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1f77b4; }
    .footer { text-align: center; color: #6c757d; padding: 20px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar
with st.sidebar:
    st.header("⚙️ Engine de Simulação")
    n_obs = st.slider("Tamanho da Amostra", 1000, 50000, 10000)
    default_rate = st.slider("Taxa de Default Alvo (%)", 1, 30, 8) / 100
    model_quality = st.slider("Qualidade (Separabilidade)", 0.1, 3.0, 1.2)
    threshold_input = st.slider("Threshold Fixo (Corte PD)", 0.0, 1.0, 0.15)

# 3. Funções de Dados e Métricas
@st.cache_data
def generate_data(n, dr, quality):
    X, y = make_classification(
        n_samples=n, n_features=15, n_informative=10, 
        n_redundant=2, weights=[1 - dr, dr], class_sep=quality, random_state=42
    )
    model = LogisticRegression()
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    return y, probs, auc

y_true, y_prob, auc_val = generate_data(n_obs, default_rate, model_quality)

def get_stats(y_true, y_prob, t_value, is_percentile=False):
    if is_percentile:
        actual_t = np.percentile(y_prob, 100 - (t_value * 100))
    else:
        actual_t = t_value
    
    preds = (y_prob >= actual_t).astype(int)
    
    # Matriz de Confusão
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    
    # Métricas Propostas
    tpr = (tp / total_pos) * 100  # % de maus que eu peguei (igual ao Recall)
    fpr = (fp / total_neg) * 100  # % de bons que eu neguei injustamente
    fnr = (fn / total_pos) * 100  # % de maus que passaram (minha falha)
    
    acc = accuracy_score(y_true, preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    boms = y_prob[y_true == 0]
    maus = y_prob[y_true == 1]
    ks = ks_2samp(boms, maus).statistic
    
    return {
        "KS": round(ks * 100, 2), "AUC": round(auc_val, 3), "Accuracy": round(acc, 3),
        "Precision": round(precision, 3), "TPR": round(tpr, 2), "FPR": round(fpr, 2), 
        "FNR": round(fnr, 2), "Threshold": round(actual_t, 4)
    }

stats_fixed = get_stats(y_true, y_prob, threshold_input)
stats_perc = get_stats(y_true, y_prob, default_rate, is_percentile=True)

# 4. Layout
st.title("🛡️ Credit Risk Model Simulator")

c1, c2, c3, c4 = st.columns(4)
c1.metric("KS do Modelo", f"{stats_fixed['KS']}%")
c2.metric("ROC AUC", f"{stats_fixed['AUC']}")
with c3:
    st.write("**Acurácia Global**")
    s1, s2 = st.columns(2); s1.metric("Fixo", f"{stats_fixed['Accuracy']*100:.1f}%"); s2.metric("Top %", f"{stats_perc['Accuracy']*100:.1f}%")
with c4:
    st.write("**TPR (Recall)**")
    s1, s2 = st.columns(2); s1.metric("Fixo", f"{stats_fixed['TPR']}%"); s2.metric("Top %", f"{stats_perc['TPR']}%")

# Gráfico Refinado
st.write("---")
fig, ax = plt.subplots(figsize=(14, 5.5))
# Cores fortes: Pure Green e Pure Red
sns.histplot(y_prob[y_true == 0], bins=60, kde=True, color='#008000', label='Bons', ax=ax, stat="density", alpha=0.3, edgecolor=None)
sns.histplot(y_prob[y_true == 1], bins=60, kde=True, color='#FF0000', label='Maus', ax=ax, stat="density", alpha=0.3, edgecolor=None)

ax.axvline(threshold_input, color='#1f77b4', linestyle='--', linewidth=2.5, label='Threshold Fixo')
ax.axvline(stats_perc['Threshold'], color='#2c3e50', linestyle=':', linewidth=2.5, label='Threshold Percentil')

ax.set_title("Distribuição de PD por Classe Real", fontsize=15, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend()
st.pyplot(fig)

# Tabela com as novas métricas de taxa
st.subheader("📊 Eficiência das Estratégias")

comparison_data = {
    "Estratégia": ["A: Threshold Fixo", "B: Top % Percentil"],
    "Threshold": [stats_fixed['Threshold'], stats_perc['Threshold']],
    "Precision": [stats_fixed['Precision'], stats_perc['Precision']],
    "Recall (Sensibilidade)": [f"{stats_fixed['TPR']}%", f"{stats_perc['TPR']}%"],
    "TPR (% de Maus Capturados)": [f"{stats_fixed['TPR']}%", f"{stats_perc['TPR']}%"],
    "FPR (% de Bons Negados)": [f"{stats_fixed['FPR']}%", f"{stats_perc['FPR']}%"],
    "FNR (% de Maus que Passaram)": [f"{stats_fixed['FNR']}%", f"{stats_perc['FNR']}%"]
}

df_comp = pd.DataFrame(comparison_data).set_index("Estratégia")

# Estilização para destacar os ganhos de cada estratégia
st.dataframe(
    df_comp.style.highlight_max(
        axis=0, 
        subset=["Precision", "Recall (Sensibilidade)", "TPR (% de Maus Capturados)"], 
        color='#e1f5fe'
    ), 
    use_container_width=True
)

# 5. Texto Relacional
st.info("""
**📖 Glossário Técnico e Relações de Risco**

* **TPR (True Positive Rate) ou Sensibilidade:** Representa a eficiência do modelo em identificar o evento de default. No gerenciamento de risco, um TPR elevado indica que a estratégia é eficaz em **mitigar perdas**, capturando a maior parte dos potenciais inadimplentes.
* **FPR (False Positive Rate) ou Taxa de Falso Alarme:** É a proporção de bons pagadores que são classificados incorretamente como maus. Esta métrica quantifica o **Custo de Oportunidade** e o potencial atrito comercial, pois representa clientes saudáveis que tiveram o crédito negado.
* **FNR (False Negative Rate) ou Taxa de Omissão:** Indica a parcela de inadimplentes que o modelo falhou em detectar. É o **Risco Residual** que efetivamente entrará para o balanço e poderá virar *Write-off* (perda real).
* **O Trade-off:** Acurácia é frequentemente enviesada pela baixa taxa de default (evento raro). O foco da gestão deve ser a otimização da **Precision** (assertividade da negativa) vs. **Recall** (cobertura do risco), onde o ponto ótimo depende do custo do capital e da margem de lucro do produto.
""")

st.markdown("<div class='footer'>AI Credit Risk Lab | 2026</div>", unsafe_allow_html=True)

# 5. Seção Educativa Expandida
with st.expander("📚 Guia Técnico: Entendendo as Métricas de Risco de Crédito"):
    st.markdown("""
    ### Matriz de Confusão no Crédito
    No contexto de Risco, nosso 'Positivo' é o evento de **Default (Mau)**.
    
    * **TP (True Positive):** O modelo disse que era Mau e o cliente realmente não pagou. É a nossa **perda evitada**.
    * **FP (False Positive):** O modelo disse que era Mau, mas o cliente era um bom pagador. É o nosso **custo de oportunidade** (perda de receita).
    * **FN (False Negative):** O modelo disse que era Bom, mas o cliente deu default. É o nosso **prejuízo real**.
    
    ### As Pergunta-Chave:
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Precision (Assertividade)**\n\n*Cálculo:* $TP / (TP + FP)$\n\n*Pergunta:* De todas as negações que o modelo sugeriu, quantas estavam certas?")
    with col_b:
        st.success("**Recall (Cobertura/Sensibilidade)**\n\n*Cálculo:* $TP / (TP + FN)$\n\n*Pergunta:* De todos os inadimplentes reais, qual % eu consegui identificar e bloquear?")

    st.markdown("""
    ### O KS (Kolmogorov-Smirnov)
    O KS avalia a **separação máxima** entre a curva acumulada de bons e maus. 
    * Independente do threshold, ele mede a "força" do modelo. 
    * Em crédito, um KS acima de 30% é considerado aceitável; acima de 50% é excelente.
    """)

st.markdown("<div class='footer'>Model Risk Management Tools | v1.1</div>", unsafe_allow_html=True)