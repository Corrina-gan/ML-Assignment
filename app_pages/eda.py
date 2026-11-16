"""Show the warehouse inspection and random-policy baseline."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eda"

st.title("Warehouse EDA")
st.caption(
    "Measured from the installed rware-tiny-2ag-v2 environment. "
    "The reward function was not changed. "
    "The baseline is 100 episodes of legal random actions, seeds 0 through 99."
)


@st.cache_data
def read_table(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str, modified: float) -> dict:
    del modified
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def read_text(path: str, modified: float) -> str:
    del modified
    return Path(path).read_text(encoding="utf-8")


def modified(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


required = (
    "statistical_summary.csv",
    "random_baseline.csv",
    "environment_inspection.json",
    "observation_action_analysis.json",
    "eda_results.json",
    "eda_report.md",
    "warehouse_layout.png",
    "sample_episode.gif",
    "action_descriptions.csv",
    "observation_components.csv",
    "representative_observations.csv",
    "blocked_movement_summary.csv",
)
missing = [name for name in required if not (RESULTS / name).exists()]

if missing:
    st.warning(
        "The analysis files are not in results/eda yet. "
        "Run `python -m eda.warehouse_eda` from the project root, "
        "or start that run here.",
        title="No saved EDA results",
    )
    if st.button("Run 100 random episodes", icon=":material/play_arrow:", key="run_eda_missing"):
        with st.status("Running the warehouse baseline...", expanded=True) as status:
            from eda.warehouse_eda import main

            main()
            status.update(label="Saved results/eda", state="complete")
        st.cache_data.clear()
        st.rerun()
else:
    summary = read_table(
        str(RESULTS / "statistical_summary.csv"),
        modified(RESULTS / "statistical_summary.csv"),
    )
    baseline = read_table(
        str(RESULTS / "random_baseline.csv"),
        modified(RESULTS / "random_baseline.csv"),
    )
    inspection = read_json(
        str(RESULTS / "environment_inspection.json"),
        modified(RESULTS / "environment_inspection.json"),
    )
    analysis = read_json(
        str(RESULTS / "observation_action_analysis.json"),
        modified(RESULTS / "observation_action_analysis.json"),
    )
    results = read_json(
        str(RESULTS / "eda_results.json"),
        modified(RESULTS / "eda_results.json"),
    )
    actions = read_table(
        str(RESULTS / "action_descriptions.csv"),
        modified(RESULTS / "action_descriptions.csv"),
    )
    components = read_table(
        str(RESULTS / "observation_components.csv"),
        modified(RESULTS / "observation_components.csv"),
    )
    observations = read_table(
        str(RESULTS / "representative_observations.csv"),
        modified(RESULTS / "representative_observations.csv"),
    )
    blocked = read_table(
        str(RESULTS / "blocked_movement_summary.csv"),
        modified(RESULTS / "blocked_movement_summary.csv"),
    )
    stats = dict(zip(summary["metric"], summary["value"]))

    def stat(name):
        return float(stats[name])

    with st.grid(4, gap="small"):
        st.metric("Mean reward", f"{stat('mean_reward'):.2f}", border=True, height="stretch")
        st.metric("Median reward", f"{stat('median_reward'):.0f}", border=True, height="stretch")
        st.metric(
            "Sample std",
            f"{stat('std_reward_sample'):.3f}",
            help="Sample standard deviation of the episode team reward, with one degree of freedom.",
            border=True,
            height="stretch",
        )
        st.metric(
            "Reward range",
            f"{stat('min_reward'):.0f} to {stat('max_reward'):.0f}",
            border=True,
            height="stretch",
        )
        st.metric(
            "Mean length",
            f"{stat('mean_episode_length'):.0f}",
            border=True,
            height="stretch",
        )
        st.metric(
            "Deliveries",
            f"{stat('total_deliveries'):.0f}",
            border=True,
            height="stretch",
        )
        st.metric(
            "Success rate",
            f"{stat('delivery_success_rate'):.0%}",
            help="Episodes with at least one delivery. A delivery is a per-robot reward of 1.",
            border=True,
            height="stretch",
        )
        st.metric(
            "Positive steps",
            f"{stat('positive_reward_steps'):.0f} / {stat('total_steps'):.0f}",
            border=True,
            height="stretch",
        )

    st.caption(
        f"{inspection['env_id']}: {inspection['grid_rows']} rows by "
        f"{inspection['grid_columns']} columns, {inspection['n_agents']} robots, "
        f"{inspection['n_shelves']} shelves, goals {inspection['goals']}. "
        f"Episodes end at {inspection['max_steps']} steps with terminated true and truncated false."
    )

section = st.segmented_control(
    "Section",
    ["Overview", "Environment", "Observations", "Actions", "Algorithms", "Report"],
    default="Overview",
    key="eda_section",
    label_visibility="collapsed",
    required=True,
)

if missing:
    st.stop()

if section == "Overview":
    reward_chart = (
        alt.Chart(baseline)
        .mark_line(point=True)
        .encode(
            x=alt.X("episode:Q", title="Episode", axis=alt.Axis(format="d")),
            y=alt.Y("reward:Q", title="Episode reward"),
            tooltip=[
                alt.Tooltip("episode:Q", format="d", title="Episode"),
                alt.Tooltip("seed:Q", format="d", title="Seed"),
                alt.Tooltip("reward:Q", format=".0f", title="Reward"),
                alt.Tooltip("deliveries:Q", format=".0f", title="Deliveries"),
                alt.Tooltip("length:Q", format=".0f", title="Length"),
            ],
        )
        .properties(title="Episode reward across the random baseline")
    )
    distribution = (
        alt.Chart(baseline)
        .mark_bar()
        .encode(
            x=alt.X("reward:Q", bin=alt.Bin(step=1), title="Episode reward"),
            y=alt.Y("count()", title="Episodes"),
            tooltip=[alt.Tooltip("count()", title="Episodes")],
        )
        .properties(title="Reward distribution")
    )
    lengths = (
        alt.Chart(baseline)
        .mark_bar()
        .encode(
            x=alt.X("length:Q", bin=alt.Bin(step=1), title="Episode length"),
            y=alt.Y("count()", title="Episodes"),
        )
        .properties(title="Episode length distribution")
    )
    outcomes = pd.DataFrame(
        {
            "Outcome": ["No delivery", "At least one delivery"],
            "Episodes": [
                int((baseline["deliveries"] == 0).sum()),
                int((baseline["deliveries"] > 0).sum()),
            ],
        }
    )
    delivery_chart = (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(
            x=alt.X("Outcome:N", title=None, sort=["No delivery", "At least one delivery"]),
            y=alt.Y("Episodes:Q", title="Episodes"),
            tooltip=["Outcome", "Episodes"],
        )
        .properties(title="Delivery performance")
    )

    delivered = int((baseline["deliveries"] > 0).sum())
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Reward")
            st.altair_chart(
                reward_chart,
                alt=f"Episode reward across {len(baseline)} random episodes",
            )
    with right:
        with st.container(border=True):
            st.subheader("Distribution")
            st.altair_chart(
                distribution,
                alt=f"Distribution of episode reward, with {delivered} episodes above zero",
            )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Length")
            st.altair_chart(
                lengths,
                alt=(
                    f"Episode length from {baseline['length'].min():.0f} "
                    f"to {baseline['length'].max():.0f} steps"
                ),
            )
            st.caption("The environment ends the episode at max_steps. A delivery does not.")
    with right:
        with st.container(border=True):
            st.subheader("Deliveries")
            st.altair_chart(
                delivery_chart,
                alt=f"{delivered} of {len(baseline)} random episodes contain a delivery",
            )

    with st.container(horizontal=True):
        st.download_button(
            "Baseline CSV",
            data=(RESULTS / "random_baseline.csv").read_bytes(),
            file_name="random_baseline.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Statistical summary",
            data=(RESULTS / "statistical_summary.csv").read_bytes(),
            file_name="statistical_summary.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "EDA report",
            data=(RESULTS / "eda_report.md").read_bytes(),
            file_name="eda_report.md",
            mime="text/markdown",
            icon=":material/download:",
        )

elif section == "Environment":
    robots = ", ".join(
        f"robot {index + 1} at ({robot['x']}, {robot['y']}) facing {robot['direction']}"
        for index, robot in enumerate(inspection["seed0_robots"])
    )
    facts = pd.DataFrame(
        [
            {"Item": "Environment", "Value": inspection["env_id"]},
            {"Item": "Grid", "Value": f"{inspection['grid_rows']} rows × {inspection['grid_columns']} columns"},
            {"Item": "Robots", "Value": inspection["n_agents"]},
            {"Item": "Shelves at reset", "Value": inspection["n_shelves"]},
            {"Item": "Storage cells", "Value": len(inspection["storage_cells"])},
            {"Item": "Shelves occupy every storage cell", "Value": inspection["shelves_match_storage"]},
            {"Item": "Goals (x, y)", "Value": str(inspection["goals"])},
            {"Item": "Goals are highways", "Value": inspection["goals_are_highways"]},
            {"Item": "Request queue size", "Value": inspection["request_queue_size"]},
            {"Item": "Seed 0 requests", "Value": str(inspection["seed0_requests"])},
            {"Item": "Seed 0 robots", "Value": robots},
            {"Item": "Sensor range", "Value": inspection["sensor_range"]},
            {"Item": "Message bits", "Value": inspection["msg_bits"]},
            {"Item": "Normalised coordinates", "Value": inspection["normalised_coordinates"]},
            {"Item": "Reward type", "Value": inspection["reward_type"]},
            {"Item": "Reward range attribute", "Value": str(inspection["reward_range"])},
            {"Item": "Max steps", "Value": inspection["max_steps"]},
            {"Item": "Max inactivity steps", "Value": str(inspection["max_inactivity_steps"])},
        ]
    )
    facts["Value"] = facts["Value"].astype(str)
    map_column, fact_column = st.columns([3, 2], vertical_alignment="top")
    with map_column:
        st.image(
            str(RESULTS / "warehouse_layout.png"),
            caption="Seed 0. Highways are light gray, shelves purple, requested shelves teal, and goals dark.",
            alt="Seed 0 warehouse with two robots, 32 shelves, and two delivery goals",
            width="stretch",
        )
        st.image(
            str(RESULTS / "sample_episode.gif"),
            caption="Seed 0 under random actions, shown every 10 steps. Circles in the path figure mark the start and squares mark the end.",
            alt="Animation of the two robots moving through the warehouse under a random policy",
            width="stretch",
        )
    with fact_column:
        st.dataframe(facts, hide_index=True, alt="Measured warehouse attributes")
        st.markdown(inspection["reward_mechanism"])
        st.markdown(inspection["termination"])
        path_image = RESULTS / "sample_episode_paths.png"
        if path_image.exists():
            st.image(
                str(path_image),
                caption="Paths of both robots during the seed 0 random episode.",
                alt="Paths of robot 1 and robot 2 across the warehouse grid",
                width="stretch",
            )

elif section == "Observations":
    observation = analysis["observation"]
    st.markdown(
        f"Space: `{observation['space']}`. "
        f"Dtype `{observation['dtype']}`, shape {observation['shape_per_robot']} per robot. "
        "Each robot gets its own vector. "
        f"The coordinates in that vector match the robot: {observation['individual_positions_match']}."
    )
    st.caption(observation["empty_cell_direction"])
    feature_group = st.segmented_control(
        "Feature group",
        ["Self", "Sensor", "All"],
        default="Self",
        key="eda_feature_group",
        required=True,
    )
    shown = components
    if feature_group == "Self":
        shown = components[components["group"] == "self"]
    elif feature_group == "Sensor":
        shown = components[components["group"] == "sensor"]
    st.dataframe(shown, hide_index=True, alt="Observation feature names and meanings")

    robot = st.segmented_control(
        "Representative observation at seed 0",
        ["Robot 1", "Robot 2"],
        default="Robot 1",
        key="eda_robot",
        required=True,
    )
    robot_number = 1 if robot == "Robot 1" else 2
    sample = observations[observations["robot"] == robot_number]
    if feature_group == "Self":
        sample = sample[sample["index"] < 8]
    elif feature_group == "Sensor":
        sample = sample[sample["index"] >= 8]
    st.dataframe(
        sample[["index", "component", "value", "meaning"]],
        hide_index=True,
        alt=f"Seed 0 observation values for robot {robot_number}",
    )

elif section == "Actions":
    st.markdown(
        f"Space: `{analysis['actions']['space']}`. "
        "Each robot chooses its own action. "
        f"There are {analysis['actions']['joint_actions']} joint action pairs."
    )
    st.dataframe(actions, hide_index=True, alt="Action ids and their effects")
    labels = {
        "forward_moved": "Forward moved",
        "forward_boundary": "Forward clamped at the edge",
        "forward_blocked_shelf": "Carried shelf blocked",
        "forward_blocked_collision": "Collision rejected",
        "toggle_pickup": "Picked up a shelf",
        "toggle_drop": "Dropped off a highway",
        "toggle_empty": "Toggle with no shelf",
        "toggle_highway_kept": "Could not drop on a highway",
        "left": "Left turn",
        "right": "Right turn",
        "noop": "No-op",
    }
    labelled = blocked[blocked["outcome"].isin(labels)].copy()
    labelled["Action outcome"] = labelled["outcome"].map(labels)
    outcome_chart = (
        alt.Chart(labelled)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Actions in the baseline"),
            y=alt.Y("Action outcome:N", sort="-x", title=None),
            tooltip=["Action outcome", "count"],
        )
        .properties(title="What the random actions did")
    )
    st.altair_chart(
        outcome_chart,
        alt="Forward moves, blocked moves, turns, and load toggles from the random baseline",
    )
    invalid = analysis["actions"]["invalid_action"]
    boundary = analysis["actions"]["boundary_example"]
    load = analysis["actions"]["load_example"]
    with st.container(border=True):
        st.subheader("Checked examples")
        st.markdown(
            f"Action id {invalid['action_id']} raised `{invalid['error']}`: {invalid['message']}."
        )
        if boundary.get("found") and boundary.get("stayed"):
            st.markdown(
                f"Robot {boundary['robot']} faced out of the grid from "
                f"({boundary['before']['x']}, {boundary['before']['y']}) and stayed there."
            )
        if load.get("found"):
            st.markdown(
                f"Robot {load['robot']} on seed {load['seed']} picked up a shelf: "
                f"{load['pickup']}. Toggle on a highway kept the load: "
                f"{load['kept_load_on_highway']}. "
                f"A later toggle in storage dropped it: {load['dropped_in_storage']}."
            )

elif section == "Algorithms":
    st.caption(
        "These notes use the measured observation, action, reward, and baseline. "
        "They do not change the environment."
    )
    for name, text in results["algorithm_notes"].items():
        with st.container(border=True):
            st.subheader(name)
            st.markdown(text)

elif section == "Report":
    st.markdown(
        read_text(str(RESULTS / "eda_report.md"), modified(RESULTS / "eda_report.md"))
    )

with st.expander("Run the analysis again"):
    st.caption("Replays 100 episodes and overwrites results/eda. This takes about a minute.")
    if st.button("Run 100 random episodes", icon=":material/play_arrow:", key="run_eda"):
        with st.status("Running the warehouse baseline...", expanded=True) as status:
            from eda.warehouse_eda import main

            main()
            status.update(label="Saved results/eda", state="complete")
        st.cache_data.clear()
        st.rerun()
