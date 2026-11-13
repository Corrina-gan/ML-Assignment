import streamlit as st

st.set_page_config(
    page_title="Warehouse robots",
    page_icon=":material/warehouse:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/warehouse.py",
            title="Warehouse",
            icon=":material/warehouse:",
            default=True,
        ),
        st.Page(
            "app_pages/algorithms.py",
            title="Algorithms",
            icon=":material/account_tree:",
        ),
    ],
    position="top",
)

page.run()
