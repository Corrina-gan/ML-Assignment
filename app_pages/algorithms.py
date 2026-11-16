"""Show PPO training and the evaluate() score. SARSA and REINFORCE are not built."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results" / "ppo"

st.title("Algorithms")
st.caption(
    "PPO trains on the shared warehouse. The episode limit is 1000 steps, "
    "and the score is the sum of the two robot rewards. The reward function was not changed. "
    "SARSA and REINFORCE are not trained yet."
)

algorithm = st.segmented_control(
    "Algorithm",
    ["PPO", "SARSA", "REINFORCE"],
    default="PPO",
    key="algorithm_choice",
    label_visibility="collapsed",
    required=True,
)


@st.cache_data
def read_table(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str, modified: float) -> dict:
    del modified
    return json.loads(Path(path).read_text(encoding="utf-8"))


def modified(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def show_unbuilt(name: str) -> None:
    with st.container(border=True):
        st.subheader(name)
        st.badge("Not built", icon=":material/pending:", color="gray")
        st.caption("This algorithm is not trained yet.")


if algorithm != "PPO":
    show_unbuilt(algorithm)
else:
    required = (
        "training_log.csv",
        "evaluation.json",
        "greedy_evaluation.json",
        "random_baseline.json",
        "config.json",
    )
    missing = [name for name in required if not (RESULTS / name).exists()]
    if missing:
        st.warning(
            "PPO results are not in results/ppo yet. "
            "From the project root, run `python -m algorithms.ppo.ppo`.",
            title="No saved PPO results",
        )
    else:
        log = read_table(str(RESULTS / "training_log.csv"), modified(RESULTS / "training_log.csv"))
        ppo = read_json(str(RESULTS / "evaluation.json"), modified(RESULTS / "evaluation.json"))
        greedy = read_json(
            str(RESULTS / "greedy_evaluation.json"),
            modified(RESULTS / "greedy_evaluation.json"),
        )
        random_score = read_json(
            str(RESULTS / "random_baseline.json"),
            modified(RESULTS / "random_baseline.json"),
        )
        config = read_json(str(RESULTS / "config.json"), modified(RESULTS / "config.json"))

        ppo_return = float(ppo["team_return_mean"])
        random_return = float(random_score["team_return_mean"])
        greedy_return = float(greedy["team_return_mean"])
        delta = ppo_return - random_return

        st.subheader("PPO")
        with st.container(horizontal=True):
            st.metric(
                "Team return",
                f"{ppo_return:.2f}",
                delta,
                delta_description="versus random, seeds 0–9",
                border=True,
                chart_data=log["team_return_mean"].tolist(),
                chart_type="line",
                height="stretch",
            )
            st.metric(
                "Mean deliveries",
                f"{float(ppo['deliveries_mean']):.2f}",
                border=True,
                height="stretch",
            )
            st.metric(
                "Greedy team return",
                f"{greedy_return:.2f}",
                border=True,
                height="stretch",
            )
            st.metric(
                "Episodes trained",
                f"{int(log['episodes'].iloc[-1])}",
                border=True,
                height="stretch",
            )

        st.caption(
            "Team return is the mean, over seeds 0 through 9, of the undiscounted sum of both robots' rewards. "
            "The PPO number samples the trained policy. "
            f"Random on the same seeds scored {random_return:.2f} "
            f"± {float(random_score['team_return_std']):.2f}. "
            "The sparkline is the team return of each training update, on seeds starting at 1000."
        )

        return_chart = (
            alt.Chart(log)
            .mark_line(point=True)
            .encode(
                x=alt.X("update:Q", title="Update"),
                y=alt.Y("team_return_mean:Q", title="Team return"),
                tooltip=[
                    alt.Tooltip("update:Q", title="Update"),
                    alt.Tooltip("episodes:Q", title="Episodes"),
                    alt.Tooltip("team_return_mean:Q", title="Team return", format=".2f"),
                    alt.Tooltip("deliveries_mean:Q", title="Deliveries", format=".2f"),
                ],
            )
        )
        baseline = (
            alt.Chart(pd.DataFrame({"team_return_mean": [random_return]}))
            .mark_rule(strokeDash=[6, 4])
            .encode(y="team_return_mean:Q")
        )
        first_return = float(log["team_return_mean"].iloc[0])
        last_return = float(log["team_return_mean"].iloc[-1])
        peak_return = float(log["team_return_mean"].max())
        st.altair_chart(
            (return_chart + baseline).properties(height=320),
            alt=(
                f"Training team return started at {first_return:.2f}, "
                f"peaked at {peak_return:.2f}, and ended at {last_return:.2f}. "
                f"The dashed line is the random score of {random_return:.2f}."
            ),
        )

        stats = log.melt(
            id_vars=["update"],
            value_vars=["entropy", "approx_kl", "policy_loss", "value_loss"],
            var_name="stat",
            value_name="value",
        )
        stat_names = {
            "entropy": "Entropy",
            "approx_kl": "Approx. KL",
            "policy_loss": "Policy loss",
            "value_loss": "Value loss",
        }
        stats["stat"] = stats["stat"].map(stat_names)
        stats_chart = (
            alt.Chart(stats)
            .mark_line()
            .encode(
                x=alt.X("update:Q", title="Update"),
                y=alt.Y("value:Q", title=None),
                tooltip=[
                    alt.Tooltip("update:Q", title="Update"),
                    alt.Tooltip("value:Q", title="Value", format=".4f"),
                ],
            )
            .properties(height=140, width=280)
            .facet(facet=alt.Facet("stat:N", title=None), columns=2)
        )
        st.altair_chart(
            stats_chart,
            alt="Policy entropy, approximate KL, and the policy and value losses across updates",
        )

        comparison = pd.DataFrame(
            [
                {
                    "Policy": "PPO, sampled",
                    "Team return": ppo_return,
                    "Std": float(ppo["team_return_std"]),
                    "Deliveries": float(ppo["deliveries_mean"]),
                    "Robot 1": float(ppo["robot_return_mean"][0]),
                    "Robot 2": float(ppo["robot_return_mean"][1]),
                },
                {
                    "Policy": "PPO, greedy",
                    "Team return": greedy_return,
                    "Std": float(greedy["team_return_std"]),
                    "Deliveries": float(greedy["deliveries_mean"]),
                    "Robot 1": float(greedy["robot_return_mean"][0]),
                    "Robot 2": float(greedy["robot_return_mean"][1]),
                },
                {
                    "Policy": "Random",
                    "Team return": random_return,
                    "Std": float(random_score["team_return_std"]),
                    "Deliveries": float(random_score["deliveries_mean"]),
                    "Robot 1": float(random_score["robot_return_mean"][0]),
                    "Robot 2": float(random_score["robot_return_mean"][1]),
                },
            ]
        )
        shown = comparison.copy()
        for column in ("Team return", "Std", "Deliveries", "Robot 1", "Robot 2"):
            shown[column] = shown[column].map(lambda value: f"{value:.2f}")
        st.dataframe(
            shown,
            hide_index=True,
            alt="PPO and random scores on seeds 0 through 9",
        )
        st.caption(
            f"{ppo['report']} "
            "Greedy actions are the actor's most likely action at each step. "
            f"Training used seeds {config['train_seeds'][0]} through {config['train_seeds'][1]}, "
            f"discount {config['gamma']}, and {config['updates']} updates of "
            f"{config['episodes_per_update']} episodes."
        )

        with st.expander("Training log"):
            st.dataframe(log, hide_index=True, alt="PPO training log, one row per update")
