import streamlit as st

st.set_page_config(
    page_title="Warehouse robots",
    page_icon=":material/warehouse:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/eda.py",
            title="EDA",
            icon=":material/query_stats:",
            default=True,
            help=(
                "Inspection of the tiny two-robot warehouse, the random-policy "
                "baseline, and what those measurements mean for SARSA, DQN, and PPO."
            ),
        ),
        st.Page(
            "app_pages/warehouse.py",
            title="Warehouse",
            icon=":material/warehouse:",
            help=(
                "Watch the two robots move in the tiny warehouse. "
                "Step, run, or reset from a chosen seed."
            ),
        ),
        st.Page(
            "app_pages/algorithms.py",
            title="Algorithms",
            icon=":material/account_tree:",
            help="PPO score on the shared warehouse. SARSA and REINFORCE are not trained yet.",
        ),
    ],
    position="top",
)

page.run()
