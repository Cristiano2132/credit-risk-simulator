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

# Função para formatar números com siglas (K, M, B)
def format_currency(value):
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"R$ {value / 1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"R$ {value / 1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"R$ {value / 1e3:.1f}K"
    else:
        return f"R$ {value:,.2f}"

# 2. Sidebar
with st.sidebar:
    st.header("⚙️ Engine de Simulação")
    n_obs = st.slider("Tamanho da Amostra", 1000, 50000, 50000)
    default_rate = st.slider("Taxa de Default Alvo (%)", 1, 30, 5) / 100
    model_quality = st.slider("Qualidade (Separabilidade)", 0.1, 3.0, 0.7)
    threshold_input = st.slider("Threshold Fixo (Corte PD)", 0.0, 1.0, 0.5)
    
    st.divider()
    st.header("💰 Matriz de Custo-Benefício")
    profit_good = st.number_input("Lucro por Bom Cliente (R$)", value=1000)
    loss_bad = st.number_input("Prejuízo por Default (R$)", value=5000)

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
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    
    tpr = (tp / total_pos) * 100 
    fpr = (fp / total_neg) * 100 
    fnr = (fn / total_pos) * 100 
    acc = accuracy_score(y_true, preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    total_pnl = (tn * profit_good) - (fn * loss_bad)
    
    boms = y_prob[y_true == 0]
    maus = y_prob[y_true == 1]
    ks = ks_2samp(boms, maus).statistic
    
    return {
        "KS": round(ks * 100, 2), "AUC": round(auc_val, 3), "Accuracy": round(acc, 3),
        "Precision": round(precision, 3), "TPR": round(tpr, 2), "FPR": round(fpr, 2), 
        "FNR": round(fnr, 2), "Threshold": round(actual_t, 4), "PnL": total_pnl
    }

stats_fixed = get_stats(y_true, y_prob, threshold_input)
stats_perc = get_stats(y_true, y_prob, default_rate, is_percentile=True)

# 4. Layout Principal
st.title("🛡️ Credit Risk Model Simulator")

# 4.1. Cards de Destaque
c1, c2, c3, c4 = st.columns(4)
c1.metric("KS do Modelo", f"{stats_fixed['KS']}%")
c2.metric("ROC AUC", f"{stats_fixed['AUC']}")
with c3:
    st.write("**Acurácia Global**")
    s1, s2 = st.columns(2); s1.metric("Fixo", f"{stats_fixed['Accuracy']*100:.1f}%"); s2.metric("Top %", f"{stats_perc['Accuracy']*100:.1f}%")
with c4:
    st.write("**TPR (Recall)**")
    s1, s2 = st.columns(2); s1.metric("Fixo", f"{stats_fixed['TPR']}%"); s2.metric("Top %", f"{stats_perc['TPR']}%")

# 4.2. Gráfico de Distribuição
st.write("---")
fig, ax = plt.subplots(figsize=(14, 5.5))
sns.histplot(y_prob[y_true == 0],
    bins=60,
    kde=False,
    color='#008000',
    label='Bons',
    ax=ax,
    stat="density",
    alpha=0.5,
    edgecolor='gray',
    linewidth=0.3)

sns.histplot(y_prob[y_true == 1],
    bins=60,
    kde=False,
    color='#FF0000',
    label='Maus',
    ax=ax,
    stat="density",
    alpha=0.5,
    edgecolor='gray',
    linewidth=0.3)
ax.axvline(threshold_input, color='#1f77b4', linestyle='--', linewidth=2.5, label='Threshold Fixo')
ax.axvline(stats_perc['Threshold'], color='#2c3e50', linestyle=':', linewidth=2.5, label='Threshold Percentil')
ax.set_title("Distribuição de PD por Classe Real", fontsize=15, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend()
st.pyplot(fig)

# 4.3. Impacto Financeiro (P&L) com Formatação K, M, B
st.divider()
st.subheader("💵 Impacto Financeiro Estimado (P&L)")
f1, f2 = st.columns([1, 2])

with f1:
    st.metric(f"Lucro Estratégia A (Threshold Fixo {stats_fixed['Threshold']})", format_currency(stats_fixed['PnL']))
    st.metric(f"Lucro Estratégia B (Top % - Threshold: {stats_perc['Threshold']})", format_currency(stats_perc['PnL']))

with f2:
    fig_fin, ax_fin = plt.subplots(figsize=(10, 4))
    strats = ['Threshold Fixo', 'Top % Percentil']
    values = [stats_fixed['PnL'], stats_perc['PnL']]
    colors = ['#3498db', '#2c3e50']
    bars = ax_fin.bar(strats, values, color=colors, alpha=0.8)
    ax_fin.set_ylabel("Lucro Esperado")
    ax_fin.spines['top'].set_visible(False); ax_fin.spines['right'].set_visible(False)
    
    # Rótulos das barras formatados
    for bar in bars:
        yval = bar.get_height()
        ax_fin.text(bar.get_x() + bar.get_width()/2., yval, format_currency(yval), 
                    ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig_fin)

# 4.4. Tabela de Performance
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
st.dataframe(df_comp.style.highlight_max(axis=0, subset=["Precision", "Recall (Sensibilidade)", "TPR (% de Maus Capturados)"], color='#e1f5fe'), use_container_width=True)

# 5. Glossário e Guia Técnico
st.info("""
**📖 Glossário Técnico e Relações de Risco**
* **TPR (True Positive Rate) ou Sensibilidade:** Eficiência do modelo em mitigar perdas.
* **FPR (False Positive Rate):** Representa o **Custo de Oportunidade** (Bons negados).
* **FNR (False Negative Rate):** Representa o **Risco Residual** (Maus que entraram).
""")

with st.expander("📚 Guia Profissional: Matriz de Confusão no Crédito"):
    st.markdown("""
    ### Interpretação Executiva
    * **TP:** Perda evitada. | **FP:** Atrito comercial. | **FN:** Prejuízo real (Write-off).
    """)

st.markdown("<div class='footer'>AI Credit Risk Lab | 2026</div>", unsafe_allow_html=True)