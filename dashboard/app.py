import locale
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except locale.Error:
        pass

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Combustível Brasil Analytics",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none;}
    </style>
""", unsafe_allow_html=True)

_PAGES = Path(__file__).parent / "pages"

pages = [
    st.Page(_PAGES / "01_Visão_Geral.py",          title="Visão Geral Nacional",     icon="📊"),
    st.Page(_PAGES / "02_Comparativo.py",           title="Comparativo por Estado",   icon="🗺️"),
    st.Page(_PAGES / "03_Presidente_Prudente.py",   title="Presidente Prudente",      icon="📍"),
    st.Page(_PAGES / "04_Previsao.py",              title="Previsão de Preço",        icon="🔮"),
    st.Page(_PAGES / "05_Cidades.py",               title="Comparativo de Cidades",   icon="🏙️"),
]

pg = st.navigation(pages)
pg.run()
