import streamlit as st

st.title("Algorithms")
st.caption(
    "SARSA, REINFORCE, and PPO are not built yet. "
    "These sections stay empty until each algorithm is added."
)

for name in ("SARSA", "REINFORCE", "PPO"):
    with st.container(border=True):
        st.subheader(name)
        st.badge("Not built", icon=":material/pending:", color="gray")
