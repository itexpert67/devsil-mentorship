import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AutoEDA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #FAFAF8 !important;
    color: #1C1C1A !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 3.5rem 6rem !important; max-width: 1200px !important; }

section[data-testid="stSidebar"] {
    background: #F5F5F2 !important;
    border-right: 1px solid #E8E8E4 !important;
}
section[data-testid="stSidebar"] * { color: #6B6B67 !important; }
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.82rem !important;
    color: #6B6B67 !important;
}

/* ── Typography ── */
.page-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #A8A8A4;
    margin-bottom: 0.6rem;
}
.page-title {
    font-family: 'Lora', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #1C1C1A;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.page-desc {
    font-size: 1rem;
    color: #6B6B67;
    line-height: 1.6;
    max-width: 600px;
    margin-bottom: 2.5rem;
}

/* ── Stat Row ── */
.stat-row {
    display: flex;
    gap: 0;
    border: 1px solid #E8E8E4;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2.5rem;
    background: #FFFFFF;
}
.stat-item {
    flex: 1;
    padding: 1.2rem 1.6rem;
    border-right: 1px solid #E8E8E4;
}
.stat-item:last-child { border-right: none; }
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #A8A8A4;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #1C1C1A;
}
.stat-value.orange { color: #D4611A; }
.stat-value.red    { color: #C0392B; }
.stat-value.green  { color: #1E7D4F; }

/* ── Section heading ── */
.sec-heading {
    font-family: 'Lora', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #1C1C1A;
    margin-top: 2.5rem;
    margin-bottom: 0.3rem;
}
.sec-sub {
    font-size: 0.85rem;
    color: #A8A8A4;
    margin-bottom: 1.2rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.03em;
}

/* ── Inline tags ── */
.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 9px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
    border: 1px solid;
}
.tag-num   { background: #EEF4FF; color: #2563EB; border-color: #BFDBFE; }
.tag-cat   { background: #FFF7ED; color: #C2410C; border-color: #FED7AA; }
.tag-bool  { background: #F0FDF4; color: #166534; border-color: #BBF7D0; }
.tag-other { background: #F5F5F2; color: #6B6B67; border-color: #E8E8E4; }

/* ── Status pills ── */
.pill-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #1E7D4F;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 0;
    font-family: 'DM Sans', sans-serif;
}
.pill-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #92400E;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 0;
    font-family: 'DM Sans', sans-serif;
}
.pill-bad {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #991B1B;
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 0;
    font-family: 'DM Sans', sans-serif;
}

/* ── Recommendation card ── */
.rec-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-left: 3px solid #D4611A;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.rec-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #D4611A;
    margin-bottom: 0.5rem;
}
.rec-card-text {
    font-size: 0.88rem;
    color: #3D3D3A;
    line-height: 1.75;
}
.rec-card-text code {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    background: #F5F5F2;
    border: 1px solid #E8E8E4;
    color: #D4611A;
    padding: 1px 6px;
    border-radius: 3px;
}
.rec-card-text strong { color: #1C1C1A; }

/* ── Model cards ── */
.model-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.8rem; margin: 1rem 0; }
.model-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    transition: border-color 0.15s;
}
.model-card:hover { border-color: #D4611A; }
.model-card-name { font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600; color: #1C1C1A; margin-bottom: 0.3rem; }
.model-card-why  { font-size: 0.78rem; color: #6B6B67; line-height: 1.55; }
.model-card-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    color: #C2410C;
    padding: 2px 7px;
    border-radius: 3px;
    margin-top: 0.5rem;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #E8E8E4; margin: 2rem 0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #E8E8E4 !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; color: #A8A8A4 !important; letter-spacing: 0.06em !important; padding: 0.7rem 1.4rem !important; background: transparent !important; border: none !important; }
.stTabs [aria-selected="true"] { color: #1C1C1A !important; border-bottom: 2px solid #D4611A !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1C1C1A !important;
    color: #FAFAF8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { background: #3D3D3A !important; }
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #1C1C1A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    border: 1px solid #E8E8E4 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
}
.stDownloadButton > button:hover { border-color: #D4611A !important; color: #D4611A !important; }

/* ── Inputs ── */
.stSelectbox > div > div { background: #FFFFFF !important; border: 1px solid #E8E8E4 !important; border-radius: 6px !important; font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important; color: #1C1C1A !important; }
.stCheckbox > label { font-family: 'DM Sans', sans-serif !important; font-size: 0.84rem !important; color: #3D3D3A !important; }
.stSlider > div { color: #3D3D3A !important; }
[data-testid="stFileUploader"] { background: #FFFFFF !important; border: 1px dashed #D0D0CC !important; border-radius: 10px !important; }
[data-testid="stFileUploader"]:hover { border-color: #D4611A !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #E8E8E4 !important; border-radius: 8px !important; }
.stDataFrame * { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #C8C8C4;
    padding: 2rem 0 1rem;
    border-top: 1px solid #E8E8E4;
    margin-top: 4rem;
}
.footer a { color: #A8A8A4; text-decoration: none; }
.footer a:hover { color: #D4611A; }
</style>
""", unsafe_allow_html=True)

# ── matplotlib light theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#E8E8E4",
    "axes.labelcolor":   "#6B6B67",
    "axes.titlecolor":   "#1C1C1A",
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.labelsize":    9,
    "xtick.color":       "#A8A8A4",
    "ytick.color":       "#A8A8A4",
    "grid.color":        "#F0F0EC",
    "grid.linewidth":    0.8,
    "text.color":        "#1C1C1A",
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ORANGE  = "#D4611A"
BLUE    = "#2563EB"
GREEN   = "#1E7D4F"
RED     = "#C0392B"
SAND    = "#E8A87C"
PALETTE = [ORANGE, BLUE, GREEN, RED, "#7C3AED", "#D97706", "#0891B2", "#DB2777"]


# ── helpers ───────────────────────────────────────────────────────────────────
def dtype_tag(dtype):
    d = str(dtype)
    if "int" in d or "float" in d: return "num"
    if "object" in d or "string" in d: return "cat"
    if "bool" in d: return "bool"
    return "other"

def tag_html(col, dtype):
    t   = dtype_tag(dtype)
    cls = {"num":"tag-num","cat":"tag-cat","bool":"tag-bool","other":"tag-other"}[t]
    return f'<span class="tag {cls}">{col}</span>'

def rec_card(label, text):
    st.markdown(
        f'<div class="rec-card">'
        f'<div class="rec-card-label">{label}</div>'
        f'<div class="rec-card-text">{text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def sec(title, sub=""):
    st.markdown(f'<div class="sec-heading">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)

def detect_problem(df):
    for col in reversed(df.columns):
        nuniq = df[col].nunique()
        dtype = str(df[col].dtype)
        if nuniq == 2: return "binary classification", col
        if "object" in dtype and nuniq <= 20: return "multiclass classification", col
        if "int" in dtype and nuniq <= 10: return "multiclass classification", col
    return "regression", df.columns[-1]


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**AutoEDA**")
    st.markdown("by Farooq Shah")
    st.markdown("---")
    uploaded     = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("---")
    show_raw     = st.checkbox("Raw data preview",       value=True)
    show_dist    = st.checkbox("Distribution plots",     value=True)
    show_corr    = st.checkbox("Correlation heatmap",    value=True)
    show_missing = st.checkbox("Missing value analysis", value=True)
    show_outlier = st.checkbox("Outlier detection",      value=True)
    st.markdown("---")
    st.markdown("farooqshah.devsil.com")

# ── page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">automated exploratory data analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">AutoEDA</div>', unsafe_allow_html=True)
st.markdown('<div class="page-desc">Upload any CSV file and get a full analysis — statistics, charts, and smart recommendations on which models to use and how to fix every issue in your data.</div>', unsafe_allow_html=True)

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    for col, title, body in [
        (c1, "Full statistics",       "Mean, median, std, skew, kurtosis — for every column, automatically."),
        (c2, "Visual analysis",       "Distributions, correlation heatmap, missing values, and outlier plots."),
        (c3, "Smart recommendations", "Which ML model fits your data, which neural network to use, and how to fix every issue detected."),
    ]:
        col.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E8E8E4;border-radius:8px;padding:1.5rem;">
          <div style="font-family:'DM Sans',sans-serif;font-weight:600;font-size:0.9rem;
                      color:#1C1C1A;margin-bottom:0.5rem;">{title}</div>
          <div style="font-size:0.82rem;color:#6B6B67;line-height:1.65;">{body}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── load ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load(f): return pd.read_csv(f)

df            = load(uploaded)
num_cols      = df.select_dtypes(include=np.number).columns.tolist()
cat_cols      = df.select_dtypes(include="object").columns.tolist()
missing_total = df.isnull().sum().sum()
dup_count     = df.duplicated().sum()
problem, target = detect_problem(df)

# ── stat bar ──────────────────────────────────────────────────────────────────
miss_cls = "red" if missing_total > 0 else "green"
st.markdown(f"""
<div class="stat-row">
  <div class="stat-item">
    <div class="stat-label">Rows</div>
    <div class="stat-value">{df.shape[0]:,}</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Columns</div>
    <div class="stat-value">{df.shape[1]}</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Numeric</div>
    <div class="stat-value orange">{len(num_cols)}</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Categorical</div>
    <div class="stat-value orange">{len(cat_cols)}</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Missing</div>
    <div class="stat-value {miss_cls}">{missing_total:,}</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Detected task</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.9rem;font-weight:500;color:#D4611A;">{problem}</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "overview", "distributions", "correlations", "missing", "outliers", "export"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sec("Column types", "each column tagged by its data type")
    st.markdown(" ".join([tag_html(c, df[c].dtype) for c in df.columns]), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("Health check")

    pills = []
    if missing_total == 0:
        pills.append('<span class="pill-ok">No missing values</span>')
    else:
        pct = round(missing_total / (df.shape[0] * df.shape[1]) * 100, 1)
        pills.append(f'<span class="pill-bad">{missing_total:,} missing values ({pct}% of dataset)</span>')

    if dup_count == 0:
        pills.append('<span class="pill-ok">No duplicate rows</span>')
    else:
        pills.append(f'<span class="pill-warn">{dup_count} duplicate rows</span>')

    if len(num_cols) > 0:
        high_skew = [c for c in num_cols if abs(df[c].skew()) > 1]
        if high_skew:
            pills.append(f'<span class="pill-warn">{len(high_skew)} highly skewed columns</span>')
        else:
            pills.append('<span class="pill-ok">No severe skewness</span>')

    st.markdown(" &nbsp; ".join(pills), unsafe_allow_html=True)

    if dup_count > 0:
        rec_card("how to fix duplicates",
            "Drop exact duplicate rows with: <code>df = df.drop_duplicates()</code><br>"
            "If duplicates are expected (same user, different dates), filter by specific columns: "
            "<code>df = df.drop_duplicates(subset=['user_id', 'date'])</code>")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("Recommended ML models", f"based on detected task: <strong>{problem}</strong>")

    if problem == "binary classification":
        models = [
            ("Logistic Regression",   "Best starting point. Fast, interpretable, works well on clean data.", "baseline"),
            ("Random Forest",         "Handles missing values, non-linear patterns, and noisy features well.", "recommended"),
            ("XGBoost / LightGBM",    "Best accuracy in most Kaggle-style tabular problems. Use after baseline.", "best for tabular"),
            ("SVM",                   "Strong on small datasets with clear margin. Slow on large data.", "small data"),
            ("Neural Network (MLP)",  "Use when dataset is large (10k+ rows) and features are dense numeric.", "deep learning"),
            ("KNN",                   "Simple and interpretable but slow at inference. Good for small data.", "simple"),
        ]
    elif problem == "multiclass classification":
        models = [
            ("Random Forest",         "Naturally handles multiple classes. Robust and interpretable.", "recommended"),
            ("XGBoost / LightGBM",    "High accuracy on structured data. Use softmax objective.", "best for tabular"),
            ("Logistic Regression",   "Use with one-vs-rest strategy. Good baseline.", "baseline"),
            ("Neural Network (MLP)",  "Strong when classes have complex boundaries and data is large.", "deep learning"),
            ("SVM (OvR)",             "Works well on medium-size datasets with many features.", "medium data"),
            ("Naive Bayes",           "Fast and good for text-based classification tasks.", "text data"),
        ]
    else:
        models = [
            ("Linear Regression",     "Always start here. Interpretable, fast, good baseline.", "baseline"),
            ("Random Forest",         "Handles non-linearity and outliers. Low tuning needed.", "recommended"),
            ("XGBoost / LightGBM",    "Best performance on structured regression problems.", "best for tabular"),
            ("Ridge / Lasso",         "Linear regression with regularization. Great when features are correlated.", "regularized"),
            ("Neural Network (MLP)",  "Use for large datasets with complex non-linear relationships.", "deep learning"),
            ("SVR",                   "Good for small-medium datasets. Needs feature scaling.", "small data"),
        ]

    cards_html = '<div class="model-row">'
    for name, why, badge in models:
        cards_html += f'<div class="model-card"><div class="model-card-name">{name}</div><div class="model-card-why">{why}</div><span class="model-card-badge">{badge}</span></div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("Recommended neural networks")

    if problem in ("binary classification", "multiclass classification"):
        nn_text = (
            "<strong>Multilayer Perceptron (MLP)</strong> — use when you have 10k+ rows and all-numeric features. "
            "Start with 2-3 hidden layers (128, 64, 32 neurons), ReLU activations, dropout 0.3.<br><br>"
            "<strong>TabNet</strong> — attention-based network designed specifically for tabular data. "
            "Outperforms MLP on structured datasets without feature engineering.<br><br>"
            "<strong>1D-CNN</strong> — if columns have a sequential or ordered relationship. "
            "Treat each row as a sequence and apply 1D convolutions.<br><br>"
            "<strong>When NOT to use neural networks</strong> — if you have less than 5,000 rows, "
            "XGBoost or Random Forest will outperform any neural network on tabular data."
        )
    else:
        nn_text = (
            "<strong>MLP Regressor</strong> — standard feed-forward network for regression. "
            "Use linear activation on output layer. Start with 3 hidden layers (256, 128, 64).<br><br>"
            "<strong>TabNet</strong> — best neural architecture for structured regression. "
            "Built-in feature selection, interpretable, strong on tabular data.<br><br>"
            "<strong>LSTM / GRU</strong> — only if your data is time-series or sequential. "
            "Not suitable for standard tabular regression.<br><br>"
            "<strong>When NOT to use neural networks</strong> — for most structured/tabular regression tasks, "
            "XGBoost with proper tuning beats neural networks unless you have 50k+ rows."
        )
    rec_card("neural network guide", nn_text)

    if show_raw:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        sec("Raw data preview")
        n = st.slider("rows to show", 5, min(200, len(df)), 10, label_visibility="collapsed")
        st.dataframe(df.head(n), use_container_width=True)

    if num_cols:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        sec("Descriptive statistics")
        desc = df[num_cols].describe().T
        desc["skewness"] = df[num_cols].skew()
        desc["kurtosis"] = df[num_cols].kurt()
        st.dataframe(
            desc.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not show_dist:
        st.info("Enable distribution plots in the sidebar.")
    elif not num_cols and not cat_cols:
        st.warning("No plottable columns found.")
    else:
        if num_cols:
            sec("Numeric distributions", "orange dashed = mean  |  green dotted = median")
            ncols_p = 3
            nrows_p = (len(num_cols) + ncols_p - 1) // ncols_p
            fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(16, 4 * nrows_p))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                data = df[col].dropna()
                axes[i].hist(data, bins=30, color=ORANGE, alpha=0.55, edgecolor="white", linewidth=0.4)
                axes[i].axvline(data.mean(),   color=ORANGE, lw=1.4, ls="--", alpha=0.9)
                axes[i].axvline(data.median(), color=GREEN,  lw=1.4, ls=":")
                axes[i].set_title(col)
                axes[i].grid(axis="y", alpha=0.5)
            for j in range(i + 1, len(axes)): axes[j].set_visible(False)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close()

            # skew recommendations
            high_skew = [(c, round(df[c].skew(), 2)) for c in num_cols if abs(df[c].skew()) > 1]
            if high_skew:
                skew_list = ", ".join([f"<strong>{c}</strong> (skew={s})" for c, s in high_skew])
                rec_card("skewness fix",
                    f"These columns have high skewness: {skew_list}<br><br>"
                    "For right-skewed (positive): apply <code>np.log1p(df[col])</code> or <code>np.sqrt(df[col])</code><br>"
                    "For left-skewed (negative): apply <code>df[col].max() - df[col]</code> then log transform<br>"
                    "In sklearn pipelines use <code>PowerTransformer(method='yeo-johnson')</code> — handles both directions automatically."
                )

        if cat_cols:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            sec("Categorical distribution")
            sel = st.selectbox("select column", cat_cols, label_visibility="collapsed")
            vc  = df[sel].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.bar(vc.index.astype(str), vc.values,
                          color=[PALETTE[i % len(PALETTE)] for i in range(len(vc))],
                          edgecolor="white", linewidth=0.5, width=0.65)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(int(bar.get_height())), ha="center", va="bottom",
                        fontsize=7.5, color="#A8A8A4")
            ax.set_title(sel)
            ax.set_xlabel(sel)
            ax.set_ylabel("count")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # class imbalance check
            top_ratio = vc.iloc[0] / vc.sum()
            if top_ratio > 0.7:
                rec_card("class imbalance detected",
                    f"The dominant class represents <strong>{top_ratio:.0%}</strong> of values in <strong>{sel}</strong> — this will bias your model.<br><br>"
                    "Options to fix:<br>"
                    "1. <strong>Oversample minority</strong>: <code>from imblearn.over_sampling import SMOTE</code><br>"
                    "2. <strong>Undersample majority</strong>: <code>RandomUnderSampler()</code><br>"
                    "3. <strong>Use class weights</strong>: <code>RandomForestClassifier(class_weight='balanced')</code><br>"
                    "4. Use <strong>F1-score or AUC</strong> instead of accuracy as your evaluation metric."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not show_corr:
        st.info("Enable correlation heatmap in the sidebar.")
    elif len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
    else:
        sec("Correlation heatmap", "pearson correlation — values range from -1 to +1")
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        size = max(8, len(num_cols) * 0.9)
        fig, ax = plt.subplots(figsize=(size, size * 0.82))
        sns.heatmap(corr, mask=mask,
                    cmap=sns.diverging_palette(15, 220, as_cmap=True),
                    annot=True, fmt=".2f", linewidths=0.5, linecolor="#FAFAF8",
                    annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.65})
        ax.set_title("Pearson Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # top pairs
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        sec("Top feature pairs")
        pairs = corr.unstack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "correlation"]
        pairs = pairs[pairs["feature_a"] < pairs["feature_b"]].sort_values("correlation", key=abs, ascending=False)
        st.dataframe(
            pairs.head(15).reset_index(drop=True)
                 .style.format({"correlation": "{:.4f}"})
                 .background_gradient(cmap="RdYlGn", subset=["correlation"]),
            use_container_width=True
        )

        # multicollinearity warning
        high_corr = pairs[pairs["correlation"].abs() > 0.85]
        if not high_corr.empty:
            pairs_list = ", ".join([f"<strong>{r.feature_a}</strong> & <strong>{r.feature_b}</strong> ({r.correlation:.2f})" for _, r in high_corr.head(3).iterrows()])
            rec_card("multicollinearity warning",
                f"Highly correlated pairs found: {pairs_list}<br><br>"
                "Multicollinearity inflates model variance and makes coefficients unreliable.<br><br>"
                "How to fix:<br>"
                "1. <strong>Drop one column</strong> from each correlated pair: <code>df.drop(columns=['col_name'])</code><br>"
                "2. <strong>PCA</strong>: reduce correlated features into uncorrelated components — <code>from sklearn.decomposition import PCA</code><br>"
                "3. <strong>Use tree models</strong> (Random Forest, XGBoost) — they handle multicollinearity naturally."
            )

        # feature selection recommendation
        if target and target in corr.columns:
            target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
            weak = target_corr[target_corr < 0.05].index.tolist()
            strong = target_corr[target_corr > 0.3].index.tolist()
            if strong:
                rec_card("feature selection",
                    f"Strongest features for <strong>{target}</strong>: {', '.join([f'<strong>{c}</strong>' for c in strong[:5]])}<br><br>"
                    + (f"Weak/uncorrelated features (consider dropping): {', '.join([f'<code>{c}</code>' for c in weak[:5]])}<br><br>" if weak else "")
                    + "Use <code>SelectKBest</code> or <code>feature_importances_</code> from a fitted Random Forest for a more rigorous selection."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not show_missing:
        st.info("Enable missing value analysis in the sidebar.")
    else:
        sec("Missing values", "by column — percentage and count")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["column", "missing_count"]
        miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
        miss = miss.sort_values("missing_count", ascending=False)

        if missing_total == 0:
            st.markdown('<span class="pill-ok">No missing values — dataset is complete</span>', unsafe_allow_html=True)
        else:
            miss_cols = miss[miss["missing_count"] > 0]
            fig, ax   = plt.subplots(figsize=(12, max(4, len(miss_cols) * 0.55)))
            colors    = [RED if p > 30 else ORANGE if p > 10 else SAND for p in miss_cols["missing_pct"]]
            ax.barh(miss_cols["column"], miss_cols["missing_pct"],
                    color=colors, edgecolor="white", linewidth=0.4, height=0.55)
            ax.axvline(10, color="#E8E8E4", lw=1, ls="--")
            ax.axvline(30, color="#FECACA", lw=1, ls="--")
            ax.set_xlabel("missing %")
            ax.set_title("Missing Values by Column")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.dataframe(miss_cols.reset_index(drop=True), use_container_width=True)

            # per-column fix recommendations
            for _, row in miss_cols.iterrows():
                col_name = row["column"]
                pct      = row["missing_pct"]
                dtype    = str(df[col_name].dtype)

                if pct > 60:
                    rec_card(f"fix — {col_name} ({pct:.1f}% missing)",
                        f"Over 60% of values are missing — <strong>drop this column</strong>: "
                        f"<code>df = df.drop(columns=['{col_name}'])</code><br>"
                        f"A column with this much missing data adds noise rather than signal."
                    )
                elif "float" in dtype or "int" in dtype:
                    rec_card(f"fix — {col_name} ({pct:.1f}% missing, numeric)",
                        f"Options for numeric imputation:<br>"
                        f"1. <strong>Median impute</strong> (best for skewed data): <code>df['{col_name}'].fillna(df['{col_name}'].median())</code><br>"
                        f"2. <strong>Mean impute</strong> (symmetric data only): <code>df['{col_name}'].fillna(df['{col_name}'].mean())</code><br>"
                        f"3. <strong>KNN impute</strong> (most accurate): <code>from sklearn.impute import KNNImputer</code><br>"
                        f"4. <strong>Add indicator column</strong>: <code>df['{col_name}_was_missing'] = df['{col_name}'].isnull().astype(int)</code> — lets the model learn from the missingness pattern."
                    )
                else:
                    rec_card(f"fix — {col_name} ({pct:.1f}% missing, categorical)",
                        f"Options for categorical imputation:<br>"
                        f"1. <strong>Mode impute</strong>: <code>df['{col_name}'].fillna(df['{col_name}'].mode()[0])</code><br>"
                        f"2. <strong>Fill with 'Unknown'</strong>: <code>df['{col_name}'].fillna('Unknown')</code><br>"
                        f"3. <strong>Use most frequent</strong> via sklearn: <code>SimpleImputer(strategy='most_frequent')</code>"
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if not show_outlier:
        st.info("Enable outlier detection in the sidebar.")
    elif not num_cols:
        st.warning("No numeric columns to analyze.")
    else:
        sec("Outlier summary", "IQR method — values beyond 1.5x the interquartile range")
        summary = []
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
            summary.append({
                "column":      col,
                "outliers":    n_out,
                "outlier_pct": round(n_out / len(df) * 100, 2),
                "lower_bound": round(lo, 3),
                "upper_bound": round(hi, 3),
            })
        out_df = pd.DataFrame(summary).sort_values("outliers", ascending=False)
        st.dataframe(
            out_df.reset_index(drop=True)
                  .style.format({"outlier_pct": "{:.2f}%"})
                  .background_gradient(cmap="RdYlGn_r", subset=["outlier_pct"]),
            use_container_width=True
        )

        # fix recommendations per column with outliers
        for _, row in out_df[out_df["outliers"] > 0].head(5).iterrows():
            col_name = row["column"]
            pct      = row["outlier_pct"]
            lo       = row["lower_bound"]
            hi       = row["upper_bound"]
            rec_card(f"fix — {col_name} ({pct:.1f}% outliers)",
                f"Choose based on your use case:<br>"
                f"1. <strong>Remove outliers</strong>: <code>df = df[(df['{col_name}'] >= {lo}) & (df['{col_name}'] <= {hi})]</code><br>"
                f"2. <strong>Cap/clip (Winsorizing)</strong>: <code>df['{col_name}'] = df['{col_name}'].clip({lo}, {hi})</code> — keeps the rows but limits extreme values<br>"
                f"3. <strong>Log transform</strong>: <code>df['{col_name}'] = np.log1p(df['{col_name}'])</code> — compresses large values naturally<br>"
                f"4. <strong>Do nothing</strong> if using tree-based models (Random Forest, XGBoost) — they are not affected by outliers."
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        sec("Box plot")
        sel_out = st.selectbox("column", num_cols, label_visibility="collapsed", key="out_sel")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        data = df[sel_out].dropna()
        axes[0].boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor="#FFF7ED", color=ORANGE),
                        whiskerprops=dict(color="#A8A8A4", lw=1.2),
                        capprops=dict(color="#A8A8A4", lw=1.2),
                        medianprops=dict(color=GREEN, lw=2),
                        flierprops=dict(marker="o", color=RED, alpha=0.5, markersize=4))
        axes[0].set_title(f"{sel_out} — box plot")
        axes[0].grid(axis="y", alpha=0.5)
        axes[1].hist(data, bins=30, color=ORANGE, alpha=0.55, edgecolor="white", linewidth=0.4)
        axes[1].set_title(f"{sel_out} — distribution")
        axes[1].grid(axis="y", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    sec("Download your data")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div style='font-size:0.82rem;color:#6B6B67;margin-bottom:0.5rem;'>Raw data</div>", unsafe_allow_html=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           file_name="eda_raw.csv", mime="text/csv", use_container_width=True)
    with c2:
        df_clean = df.drop_duplicates()
        removed  = len(df) - len(df_clean)
        st.markdown(f"<div style='font-size:0.82rem;color:#6B6B67;margin-bottom:0.5rem;'>Cleaned data <span style='color:#A8A8A4;'>({removed} duplicates removed)</span></div>", unsafe_allow_html=True)
        st.download_button("Download Cleaned CSV", df_clean.to_csv(index=False).encode(),
                           file_name="eda_cleaned.csv", mime="text/csv", use_container_width=True)
    with c3:
        if num_cols:
            st.markdown("<div style='font-size:0.82rem;color:#6B6B67;margin-bottom:0.5rem;'>Stats summary</div>", unsafe_allow_html=True)
            st.download_button("Download Stats CSV", df[num_cols].describe().to_csv().encode(),
                               file_name="eda_stats.csv", mime="text/csv", use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  built by <a href="https://farooqshah.devsil.com">farooq shah</a>
  &nbsp;&mdash;&nbsp; data scientist &amp; ml engineer &nbsp;&mdash;&nbsp; abbottabad, pakistan
</div>
""", unsafe_allow_html=True)
