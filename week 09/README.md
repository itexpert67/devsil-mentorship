# AutoEDA — Automated Exploratory Data Analysis Tool

**Author:** Farooq Shah  
**Organization:** DEVSIL (SMC-PRIVATE) LIMITED  
**Project Type:** Data Analytics + Streamlit Web App  
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://farooq-eda-tool.streamlit.app)

https://farooq-eda-tool.streamlit.app

---

## Overview

AutoEDA is a fully automated Exploratory Data Analysis tool built with Streamlit. Upload any CSV file and get instant, structured analytics — no code required.

Built for data scientists, analysts, and ML engineers who need to understand their data fast before jumping into modeling.

---

## Features

| Feature | Description |
|---------|-------------|
| Overview | Shape, dtypes, memory usage, health check for missing values and duplicates |
| Distributions | Histograms for all numeric columns, bar charts for categorical |
| Correlations | Pearson heatmap and ranked top feature pairs |
| Missing Values | Visual breakdown of null counts and percentages per column |
| Outlier Detection | IQR-based outlier detection with box plots |
| Export | Download raw, cleaned, or stats summary as CSV |

---

## Project Structure

```
AutoEDA/
├── graphs and models/          sample output charts
├── auto_eda_app.py             Streamlit app (main file)
├── AutoEDA_Notebook.ipynb      Jupyter notebook walkthrough
├── AutoEDA_Report.docx         project documentation
├── requirements.txt            Python dependencies
└── README.md                   this file
```

---

## Tech Stack

- **Streamlit** — web app framework
- **Pandas** — data manipulation
- **NumPy** — numerical computing
- **Matplotlib** — custom visualizations
- **Seaborn** — correlation heatmap

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/AutoEDA.git
cd AutoEDA
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run auto_eda_app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, set main file as `auto_eda_app.py`
4. Click Deploy — live in under 2 minutes

---

## requirements.txt

```
streamlit
pandas
numpy
matplotlib
seaborn
openpyxl
```

---

## Author

**Farooq Shah**  
Data Scientist & ML Engineer  
hayatshahm15@gmail.com  
[farooqshah.devsil.com](https://farooqshah.devsil.com)  
Abbottabad, Pakistan

---

## License

MIT License — free to use and modify.
