import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scraper import fetch_articles
from nlp import analyze

st.set_page_config(page_title="Dawn Intelligence", layout="wide", page_icon="")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }

        .dashboard-header {
            border-bottom: 1px solid #222;
            padding-bottom: 1.2rem;
            margin-bottom: 1.8rem;
        }

        .dashboard-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f5f5f5;
            letter-spacing: -0.3px;
            margin: 0;
        }

        .dashboard-sub {
            font-size: 0.8rem;
            color: #555;
            margin-top: 4px;
            font-weight: 400;
            letter-spacing: 0.3px;
        }

        .section-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #444;
            margin-bottom: 1rem;
            margin-top: 2rem;
        }

        div[data-testid="metric-container"] {
            background: #111;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 1rem 1.2rem;
        }

        div[data-testid="metric-container"] label {
            font-size: 0.7rem;
            color: #555;
            font-weight: 500;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f0f0f0;
        }

        .trend-pill {
            display: inline-block;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.72rem;
            color: #aaa;
            margin: 3px;
            font-weight: 500;
        }

        .trend-pill span {
            color: #e05a00;
            font-weight: 700;
            margin-right: 5px;
        }

        .breaking-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }

        .breaking-item {
            background: #140a00;
            border: 1px solid #3a1a00;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 0.75rem;
            color: #ff8c42;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .breaking-item .count {
            color: #555;
            font-weight: 400;
            margin-left: 8px;
            font-size: 0.7rem;
        }

        .stDataFrame {
            border: 1px solid #1e1e1e;
            border-radius: 8px;
        }

        [data-testid="stSidebar"] {
            background: #0d0d0d;
            border-right: 1px solid #1a1a1a;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 1.5rem;
        }

        hr {
            border-color: #1a1a1a;
            margin: 1.5rem 0;
        }

        .topic-card {
            background: #111;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }

        .topic-name {
            font-size: 0.72rem;
            font-weight: 600;
            color: #888;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .topic-words {
            font-size: 0.82rem;
            color: #ccc;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#aaa", size=11),
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(gridcolor="#1a1a1a", linecolor="#222", tickcolor="#333"),
    yaxis=dict(gridcolor="#1a1a1a", linecolor="#222", tickcolor="#333"),
)

COLOR_MAP = {
    "Positive": "#22c55e",
    "Neutral":  "#6b7280",
    "Negative": "#ef4444",
}

COLOR_SEQ = ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f97316", "#14b8a6"]


st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Dawn News Intelligence</div>
        <div class="dashboard-sub">LIVE NLP ANALYSIS &nbsp;·&nbsp; SENTIMENT &nbsp;·&nbsp; TOPICS &nbsp;·&nbsp; ENTITIES &nbsp;·&nbsp; TRENDS</div>
    </div>
""", unsafe_allow_html=True)


with st.spinner(""):
    df = fetch_articles()
    df, topics, people, places, orgs, trending, breaking = analyze(df)


st.sidebar.markdown("### Filters")
st.sidebar.markdown("---")

categories = ["All"] + sorted(df["category"].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)
selected_sentiment = st.sidebar.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
selected_topic = st.sidebar.selectbox("Topic", ["All"] + list(topics.keys())) if topics else "All"

filtered = df.copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment"] == selected_sentiment]
if selected_topic != "All" and "topic" in filtered.columns:
    filtered = filtered[filtered["topic"] == selected_topic]


if breaking:
    items_html = "".join([
        f'<div class="breaking-item">{s["keyword"].upper()}<span class="count">{s["mentions"]} mentions</span></div>'
        for s in breaking[:6]
    ])
    st.markdown(f'<div class="breaking-row">{items_html}</div>', unsafe_allow_html=True)


total = len(filtered)
positive = len(filtered[filtered["sentiment"] == "Positive"])
neutral = len(filtered[filtered["sentiment"] == "Neutral"])
negative = len(filtered[filtered["sentiment"] == "Negative"])
avg_polarity = round(filtered["polarity"].mean(), 3) if total > 0 else 0
avg_subjectivity = round(filtered["subjectivity"].mean(), 3) if total > 0 else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Articles", total)
c2.metric("Positive", positive)
c3.metric("Neutral", neutral)
c4.metric("Negative", negative)
c5.metric("Avg Polarity", avg_polarity)
c6.metric("Subjectivity", avg_subjectivity)

st.markdown("---")


st.markdown('<div class="section-label">Sentiment Analysis</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    if total > 0:
        counts = filtered["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = go.Figure(go.Pie(
            labels=counts["Sentiment"],
            values=counts["Count"],
            hole=0.6,
            marker=dict(colors=[COLOR_MAP.get(s, "#888") for s in counts["Sentiment"]],
                        line=dict(color="#000", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#aaa"),
        ))
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, height=240)
        st.plotly_chart(fig, width="stretch")

with col2:
    if total > 0:
        daily = filtered.copy()
        daily["date_only"] = daily["date"].dt.date
        daily_counts = daily.groupby(["date_only", "sentiment"]).size().reset_index(name="count")
        fig2 = px.line(daily_counts, x="date_only", y="count", color="sentiment",
                       color_discrete_map=COLOR_MAP,
                       labels={"date_only": "", "count": "Articles", "sentiment": ""})
        fig2.update_traces(line=dict(width=2))
        fig2.update_layout(**PLOT_LAYOUT, height=240,
                           legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, width="stretch")

st.markdown("---")


st.markdown('<div class="section-label">Category Breakdown</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    cat_sent = filtered.groupby(["category", "sentiment"]).size().reset_index(name="count")
    fig3 = px.bar(cat_sent, x="category", y="count", color="sentiment",
                  color_discrete_map=COLOR_MAP, barmode="stack",
                  labels={"category": "", "count": "Articles", "sentiment": ""})
    fig3.update_layout(**PLOT_LAYOUT, height=260,
                       legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, width="stretch")

with col4:
    fig4 = px.histogram(filtered, x="polarity", nbins=40,
                        color_discrete_sequence=["#3b82f6"],
                        labels={"polarity": "Polarity Score", "count": "Articles"})
    fig4.update_layout(**PLOT_LAYOUT, height=260, showlegend=False)
    st.plotly_chart(fig4, width="stretch")

st.markdown("---")


st.markdown('<div class="section-label">Trending Keywords</div>', unsafe_allow_html=True)
if trending:
    pills_html = "".join([
        f'<div class="trend-pill"><span>#{i+1}</span>{w}</div>'
        for i, (w, _) in enumerate(trending[:20])
    ])
    st.markdown(pills_html, unsafe_allow_html=True)

st.markdown("---")


st.markdown('<div class="section-label">Narrative Shift by Category</div>', unsafe_allow_html=True)
if total > 0:
    shift = filtered.copy()
    shift["date_only"] = shift["date"].dt.date
    shift_daily = shift.groupby(["date_only", "category"])["polarity"].mean().reset_index()
    fig5 = px.line(shift_daily, x="date_only", y="polarity", color="category",
                   color_discrete_sequence=COLOR_SEQ,
                   labels={"date_only": "", "polarity": "Avg Polarity", "category": ""})
    fig5.add_hline(y=0, line_dash="dot", line_color="#333", line_width=1)
    fig5.update_traces(line=dict(width=2))
    fig5.update_layout(**PLOT_LAYOUT, height=260,
                       legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5, width="stretch")

st.markdown("---")


st.markdown('<div class="section-label">Named Entities</div>', unsafe_allow_html=True)
e1, e2, e3 = st.columns(3)

def entity_chart(data, label, color):
    if data:
        edf = pd.DataFrame(list(data.items()), columns=[label, "Mentions"])
        fig = px.bar(edf.head(10), x="Mentions", y=label, orientation="h",
                     color="Mentions", color_continuous_scale=color)
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", gridcolor="#1a1a1a",
                                     linecolor="#222", tickcolor="#333"))
        return fig
    return None

with e1:
    st.caption("People")
    fig = entity_chart(people, "Person", "Blues")
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Requires spacy model")

with e2:
    st.caption("Places")
    fig = entity_chart(places, "Place", "Greens")
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Requires spacy model")

with e3:
    st.caption("Organizations")
    fig = entity_chart(orgs, "Organization", "Oranges")
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Requires spacy model")

st.markdown("---")


if topics:
    st.markdown('<div class="section-label">Topic Modeling</div>', unsafe_allow_html=True)

    t_cols = st.columns(3)
    for i, (topic, words) in enumerate(topics.items()):
        with t_cols[i % 3]:
            st.markdown(f"""
                <div class="topic-card">
                    <div class="topic-name">{topic}</div>
                    <div class="topic-words">{" &nbsp;·&nbsp; ".join(words)}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    wc_topic = st.selectbox("Word cloud topic", list(topics.keys()), label_visibility="collapsed")
    topic_text = " ".join(topics[wc_topic] * 20)
    wc = WordCloud(
        width=1400, height=380,
        background_color="#0d0d0d",
        colormap="Blues",
        max_font_size=110,
        min_font_size=14,
    ).generate(topic_text)
    fig_wc, ax = plt.subplots(figsize=(14, 4))
    fig_wc.patch.set_facecolor("#0d0d0d")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

st.markdown("---")


st.markdown('<div class="section-label">Article Feed</div>', unsafe_allow_html=True)
display_cols = ["date", "category", "sentiment", "polarity", "subjectivity", "topic", "title"]
display_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[display_cols].reset_index(drop=True), width="stretch", hide_index=True)
