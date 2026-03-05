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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #09090B !important;
    color: #D4D4D8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 3rem 5rem !important; max-width: 1380px !important; }

section[data-testid="stSidebar"] {
    background: #0F0F11 !important;
    border-right: 1px solid #1F1F23 !important;
}
section[data-testid="stSidebar"] * { color: #A1A1AA !important; }
section[data-testid="stSidebar"] label {
    color: #71717A !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    color: #FAFAFA;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.page-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #3F3F46;
    letter-spacing: 0.08em;
    margin-bottom: 2.5rem;
}
.page-sub span { color: #E97316; }

.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1F1F23;
    border: 1px solid #1F1F23;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.stat-cell { background: #09090B; padding: 1.4rem 1.6rem; }
.stat-cell:hover { background: #0F0F11; }
.stat-cell-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3F3F46;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.stat-cell-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #FAFAFA;
}
.stat-cell-value.accent { color: #E97316; }
.stat-cell-value.warn   { color: #EF4444; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3F3F46;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1F1F23;
}

.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin: 2px 3px 2px 0;
}
.tag-num   { background: #1C2333; color: #93C5FD; border: 1px solid #1E3A5F; }
.tag-cat   { background: #1C1810; color: #FCD34D; border: 1px solid #3D2E00; }
.tag-bool  { background: #0F1F15; color: #6EE7B7; border: 1px solid #064E3B; }
.tag-other { background: #18181B; color: #71717A;  border: 1px solid #27272A; }

.health-ok {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #6EE7B7;
    background: #0F1F15;
    border: 1px solid #064E3B;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
}
.health-warn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #FCA5A5;
    background: #1C0F0F;
    border: 1px solid #7F1D1D;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
}

.stButton > button {
    background: #E97316 !important;
    color: #09090B !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { opacity: 0.8 !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: #A1A1AA !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    border: 1px solid #27272A !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
}
.stDownloadButton > button:hover { border-color: #E97316 !important; color: #E97316 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1F1F23 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #3F3F46 !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1.4rem !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #FAFAFA !important;
    border-bottom: 1px solid #E97316 !important;
}

.stDataFrame { border: 1px solid #1F1F23 !important; border-radius: 8px !important; }
.stDataFrame * { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }

[data-testid="stFileUploader"] {
    background: #0F0F11 !important;
    border: 1px dashed #27272A !important;
    border-radius: 10px !important;
}

.stSelectbox > div > div {
    background: #0F0F11 !important;
    border: 1px solid #27272A !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stCheckbox > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

.foot {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #27272A;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1F1F23;
    margin-top: 4rem;
}
.foot a { color: #3F3F46; text-decoration: none; }
.foot a:hover { color: #E97316; }
</style>
""", unsafe_allow_html=True)

# -- matplotlib theme ----------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#09090B",
    "axes.facecolor":    "#09090B",
    "axes.edgecolor":    "#1F1F23",
    "axes.labelcolor":   "#52525B",
    "axes.titlecolor":   "#D4D4D8",
    "axes.titlesize":    11,
    "axes.titleweight":  "semibold",
    "axes.titlepad":     12,
    "axes.labelsize":    9,
    "xtick.color":       "#3F3F46",
    "ytick.color":       "#3F3F46",
    "grid.color":        "#1F1F23",
    "grid.linewidth":    0.6,
    "text.color":        "#D4D4D8",
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ORANGE  = "#E97316"
BLUE    = "#60A5FA"
GREEN   = "#34D399"
RED     = "#F87171"
PALETTE = [ORANGE, BLUE, GREEN, RED, "#A78BFA", "#FBBF24", "#2DD4BF", "#F472B6"]


def dtype_tag(dtype):
    d = str(dtype)
    if "int" in d or "float" in d: return "num"
    if "object" in d or "string" in d: return "cat"
    if "bool" in d: return "bool"
    return "other"


def tag_html(col, dtype):
    t   = dtype_tag(dtype)
    cls = {"num": "tag-num", "cat": "tag-cat", "bool": "tag-bool", "other": "tag-other"}[t]
    return f'<span class="tag {cls}">{col}</span>'


# -- sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:1rem;"
        "font-weight:600;color:#FAFAFA;margin-bottom:0.3rem;'>AutoEDA</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
        "color:#3F3F46;margin-bottom:1.5rem;'>farooq shah</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("<hr style='border-color:#1F1F23;margin:1.2rem 0;'>", unsafe_allow_html=True)
    show_raw     = st.checkbox("Raw data preview",        value=True)
    show_dist    = st.checkbox("Distribution plots",      value=True)
    show_corr    = st.checkbox("Correlation heatmap",     value=True)
    show_missing = st.checkbox("Missing value analysis",  value=True)
    show_outlier = st.checkbox("Outlier detection",       value=True)
    st.markdown("<hr style='border-color:#1F1F23;margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
        "color:#27272A;'>farooqshah.devsil.com</div>",
        unsafe_allow_html=True
    )

# -- header --------------------------------------------------------------------
st.markdown("""
<div class="page-title">AutoEDA</div>
<div class="page-sub">Automated Exploratory Data Analysis &mdash; by <span>Farooq Shah</span></div>
""", unsafe_allow_html=True)

# -- empty state ---------------------------------------------------------------
if uploaded is None:
    st.markdown("<div class='section-label'>what this does</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "Full Statistics",
             "Mean, median, std, skew, kurtosis for every numeric column automatically."),
        (c2, "Visual Analysis",
             "Distributions, correlation heatmap, outlier box plots — rendered instantly."),
        (c3, "Export Ready",
             "Download cleaned data and stats summary CSV for your next ML project."),
    ]:
        col.markdown(f"""
        <div style="background:#0F0F11;border:1px solid #1F1F23;
                    border-radius:8px;padding:1.6rem;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                      font-weight:600;color:#FAFAFA;margin-bottom:0.6rem;">{title}</div>
          <div style="font-size:0.82rem;color:#52525B;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# -- load data -----------------------------------------------------------------
@st.cache_data
def load(f):
    return pd.read_csv(f)


df            = load(uploaded)
num_cols      = df.select_dtypes(include=np.number).columns.tolist()
cat_cols      = df.select_dtypes(include="object").columns.tolist()
missing_total = df.isnull().sum().sum()
dup_count     = df.duplicated().sum()

# -- stat grid -----------------------------------------------------------------
miss_cls = "warn" if missing_total > 0 else "accent"
st.markdown(f"""
<div class="stat-grid">
  <div class="stat-cell">
    <div class="stat-cell-label">Rows</div>
    <div class="stat-cell-value">{df.shape[0]:,}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Columns</div>
    <div class="stat-cell-value">{df.shape[1]}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Numeric</div>
    <div class="stat-cell-value accent">{len(num_cols)}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Categorical</div>
    <div class="stat-cell-value accent">{len(cat_cols)}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Missing</div>
    <div class="stat-cell-value {miss_cls}">{missing_total:,}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -- tabs ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "overview", "distributions", "correlations", "missing", "outliers", "export"
])

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-label' style='margin-top:1.5rem;'>column types</div>",
                unsafe_allow_html=True)
    st.markdown(
        " ".join([tag_html(c, df[c].dtype) for c in df.columns]),
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-label' style='margin-top:2rem;'>health check</div>",
                unsafe_allow_html=True)
    if missing_total == 0:
        st.markdown(
            '<div class="health-ok">no missing values — dataset is complete</div>',
            unsafe_allow_html=True
        )
    else:
        pct = round(missing_total / (df.shape[0] * df.shape[1]) * 100, 1)
        st.markdown(
            f'<div class="health-warn">{missing_total:,} missing values — {pct}% of all cells</div>',
            unsafe_allow_html=True
        )

    msg = "no duplicate rows found" if dup_count == 0 else f"{dup_count} duplicate rows detected"
    cls = "health-ok" if dup_count == 0 else "health-warn"
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    if show_raw:
        st.markdown("<div class='section-label' style='margin-top:2rem;'>raw data preview</div>",
                    unsafe_allow_html=True)
        n = st.slider("rows", 5, min(200, len(df)), 10, label_visibility="collapsed")
        st.dataframe(df.head(n), use_container_width=True)

    if num_cols:
        st.markdown("<div class='section-label' style='margin-top:2rem;'>descriptive statistics</div>",
                    unsafe_allow_html=True)
        desc = df[num_cols].describe().T
        desc["skewness"] = df[num_cols].skew()
        desc["kurtosis"] = df[num_cols].kurt()
        st.dataframe(
            desc.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True
        )

# ── DISTRIBUTIONS ─────────────────────────────────────────────────────────────
with tab2:
    if not show_dist:
        st.markdown('<div class="health-warn">enable distribution plots in the sidebar</div>',
                    unsafe_allow_html=True)
    elif not num_cols and not cat_cols:
        st.markdown('<div class="health-warn">no plottable columns found</div>',
                    unsafe_allow_html=True)
    else:
        if num_cols:
            st.markdown("<div class='section-label' style='margin-top:1.5rem;'>numeric distributions</div>",
                        unsafe_allow_html=True)
            ncols_plot = 3
            nrows_plot = (len(num_cols) + ncols_plot - 1) // ncols_plot
            fig, axes  = plt.subplots(nrows_plot, ncols_plot,
                                      figsize=(16, 4.2 * nrows_plot))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                ax   = axes[i]
                data = df[col].dropna()
                ax.hist(data, bins=30, color=ORANGE, alpha=0.65, edgecolor="none")
                ax.axvline(data.mean(),   color=BLUE,  lw=1.2, ls="--")
                ax.axvline(data.median(), color=GREEN, lw=1.2, ls=":")
                ax.set_title(col)
                ax.grid(axis="y", alpha=0.4)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close()

        if cat_cols:
            st.markdown("<div class='section-label' style='margin-top:2rem;'>categorical distribution</div>",
                        unsafe_allow_html=True)
            sel = st.selectbox("column", cat_cols, label_visibility="collapsed")
            vc  = df[sel].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.bar(
                vc.index.astype(str), vc.values,
                color=[PALETTE[i % len(PALETTE)] for i in range(len(vc))],
                edgecolor="none", width=0.6
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=7.5, color="#52525B"
                )
            ax.set_title(sel)
            ax.set_xlabel(sel)
            ax.set_ylabel("count")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── CORRELATIONS ──────────────────────────────────────────────────────────────
with tab3:
    if not show_corr:
        st.markdown('<div class="health-warn">enable correlation heatmap in the sidebar</div>',
                    unsafe_allow_html=True)
    elif len(num_cols) < 2:
        st.markdown('<div class="health-warn">need at least 2 numeric columns</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-label' style='margin-top:1.5rem;'>pearson correlation matrix</div>",
                    unsafe_allow_html=True)
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        size = max(8, len(num_cols) * 0.9)
        fig, ax = plt.subplots(figsize=(size, size * 0.82))
        sns.heatmap(
            corr, mask=mask,
            cmap=sns.diverging_palette(220, 15, as_cmap=True),
            annot=True, fmt=".2f",
            linewidths=0.4, linecolor="#09090B",
            annot_kws={"size": 8}, ax=ax,
            cbar_kws={"shrink": 0.65}
        )
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='section-label' style='margin-top:2rem;'>top feature pairs</div>",
                    unsafe_allow_html=True)
        pairs = corr.unstack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "correlation"]
        pairs = (pairs[pairs["feature_a"] < pairs["feature_b"]]
                 .sort_values("correlation", key=abs, ascending=False))
        st.dataframe(
            pairs.head(15).reset_index(drop=True)
                 .style.format({"correlation": "{:.4f}"})
                 .background_gradient(cmap="RdYlGn", subset=["correlation"]),
            use_container_width=True
        )

# ── MISSING VALUES ────────────────────────────────────────────────────────────
with tab4:
    if not show_missing:
        st.markdown('<div class="health-warn">enable missing value analysis in the sidebar</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-label' style='margin-top:1.5rem;'>missing values by column</div>",
                    unsafe_allow_html=True)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["column", "missing_count"]
        miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
        miss = miss.sort_values("missing_count", ascending=False)

        if missing_total == 0:
            st.markdown(
                '<div class="health-ok">no missing values — nothing to show here</div>',
                unsafe_allow_html=True
            )
        else:
            miss_cols = miss[miss["missing_count"] > 0]
            fig, ax   = plt.subplots(figsize=(12, max(4, len(miss_cols) * 0.55)))
            colors    = [RED if p > 10 else ORANGE for p in miss_cols["missing_pct"]]
            ax.barh(miss_cols["column"], miss_cols["missing_pct"],
                    color=colors, edgecolor="none", height=0.55)
            ax.axvline(10, color="#27272A", lw=0.8, ls="--")
            ax.set_xlabel("missing %")
            ax.set_title("Missing Values")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(miss_cols.reset_index(drop=True), use_container_width=True)

# ── OUTLIERS ──────────────────────────────────────────────────────────────────
with tab5:
    if not show_outlier:
        st.markdown('<div class="health-warn">enable outlier detection in the sidebar</div>',
                    unsafe_allow_html=True)
    elif not num_cols:
        st.markdown('<div class="health-warn">no numeric columns to analyze</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-label' style='margin-top:1.5rem;'>iqr outlier summary</div>",
                    unsafe_allow_html=True)
        summary = []
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
            summary.append({
                "column":       col,
                "outliers":     n_out,
                "outlier_pct":  round(n_out / len(df) * 100, 2),
                "lower_bound":  round(lo, 3),
                "upper_bound":  round(hi, 3),
            })
        out_df = pd.DataFrame(summary).sort_values("outliers", ascending=False)
        st.dataframe(
            out_df.reset_index(drop=True)
                  .style.format({"outlier_pct": "{:.2f}%"})
                  .background_gradient(cmap="Reds", subset=["outlier_pct"]),
            use_container_width=True
        )

        st.markdown("<div class='section-label' style='margin-top:2rem;'>box plot</div>",
                    unsafe_allow_html=True)
        sel_out = st.selectbox("column", num_cols, label_visibility="collapsed", key="out_sel")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        data = df[sel_out].dropna()
        axes[0].boxplot(
            data, patch_artist=True,
            boxprops=dict(facecolor="#1F1F23", color=ORANGE),
            whiskerprops=dict(color=BLUE, lw=1.2),
            capprops=dict(color=BLUE, lw=1.2),
            medianprops=dict(color=GREEN, lw=2),
            flierprops=dict(marker="o", color=ORANGE, alpha=0.35, markersize=3.5)
        )
        axes[0].set_title(f"{sel_out} — box plot")
        axes[0].grid(axis="y", alpha=0.4)
        axes[1].hist(data, bins=30, color=BLUE, alpha=0.65, edgecolor="none")
        axes[1].set_title(f"{sel_out} — distribution")
        axes[1].grid(axis="y", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── EXPORT ────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-label' style='margin-top:1.5rem;'>download</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "<div style='font-size:0.78rem;color:#52525B;margin-bottom:0.6rem;"
            "font-family:IBM Plex Mono,monospace;'>raw data</div>",
            unsafe_allow_html=True
        )
        st.download_button(
            "download csv",
            df.to_csv(index=False).encode(),
            file_name="eda_raw.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:
        df_clean  = df.drop_duplicates()
        removed   = len(df) - len(df_clean)
        st.markdown(
            f"<div style='font-size:0.78rem;color:#52525B;margin-bottom:0.6rem;"
            f"font-family:IBM Plex Mono,monospace;'>cleaned data "
            f"<span style='color:#3F3F46;'>({removed} dupes removed)</span></div>",
            unsafe_allow_html=True
        )
        st.download_button(
            "download cleaned csv",
            df_clean.to_csv(index=False).encode(),
            file_name="eda_cleaned.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c3:
        if num_cols:
            st.markdown(
                "<div style='font-size:0.78rem;color:#52525B;margin-bottom:0.6rem;"
                "font-family:IBM Plex Mono,monospace;'>stats summary</div>",
                unsafe_allow_html=True
            )
            st.download_button(
                "download stats csv",
                df[num_cols].describe().to_csv().encode(),
                file_name="eda_stats.csv",
                mime="text/csv",
                use_container_width=True
            )

# -- footer --------------------------------------------------------------------
st.markdown("""
<div class="foot">
  built by <a href="https://farooqshah.devsil.com">farooq shah</a>
  &nbsp;&mdash;&nbsp; data scientist &amp; ml engineer &nbsp;&mdash;&nbsp; abbottabad, pakistan
</div>
""", unsafe_allow_html=True)
