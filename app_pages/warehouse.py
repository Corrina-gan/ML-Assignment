import time

import gymnasium as gym
import numpy as np
import rware
import streamlit as st
from rware.warehouse import Direction

ENV_ID = "rware-tiny-2ag-v2"
CELL = 40
ACTION_NAMES = ("No-op", "Forward", "Left", "Right", "Toggle load")
ROBOT_COLORS = (
    (255, 140, 0),
    (30, 136, 229),
    (67, 160, 71),
    (142, 36, 170),
)

FLOOR = (248, 248, 248)
AISLE = (230, 230, 230)
GRID = (190, 190, 190)
SHELF = (72, 61, 139)
REQUESTED = (0, 128, 128)
GOAL = (60, 60, 60)
CARRY = (255, 255, 255)
FACING = (20, 20, 20)


def open_warehouse(seed):
    previous = st.session_state.get("warehouse_env")
    if previous is not None:
        previous.close()

    env = gym.make(ENV_ID)
    observation, info = env.reset(seed=int(seed))
    st.session_state.warehouse_env = env
    st.session_state.warehouse_obs = observation
    st.session_state.warehouse_step = 0
    st.session_state.warehouse_episode = 1
    st.session_state.warehouse_return = 0.0
    st.session_state.warehouse_last_actions = None
    st.session_state.warehouse_last_reward = None
    st.session_state.warehouse_done = False
    return info


def begin_next_episode():
    env = st.session_state.warehouse_env
    observation, info = env.reset()
    st.session_state.warehouse_obs = observation
    st.session_state.warehouse_step = 0
    st.session_state.warehouse_episode += 1
    st.session_state.warehouse_return = 0.0
    st.session_state.warehouse_last_actions = None
    st.session_state.warehouse_last_reward = None
    st.session_state.warehouse_done = False
    return info


def choose_actions(env):
    # The existing environment samples a legal action for every robot.
    # SARSA, REINFORCE, and PPO are not connected yet.
    return env.action_space.sample()


def step_warehouse():
    if st.session_state.warehouse_done:
        begin_next_episode()

    env = st.session_state.warehouse_env
    actions = choose_actions(env)
    observation, reward, terminated, truncated, info = env.step(actions)
    reward = [float(value) for value in reward]

    st.session_state.warehouse_obs = observation
    st.session_state.warehouse_last_actions = tuple(int(action) for action in actions)
    st.session_state.warehouse_last_reward = reward
    st.session_state.warehouse_return += float(sum(reward))
    st.session_state.warehouse_step += 1
    st.session_state.warehouse_done = bool(terminated or truncated)
    return info


def _fill_cell(image, x, y, color, pad):
    y0 = y * CELL + pad
    x0 = x * CELL + pad
    y1 = (y + 1) * CELL - pad
    x1 = (x + 1) * CELL - pad
    image[y0:y1, x0:x1] = color


def _paint_facing(image, x, y, direction):
    y0 = y * CELL + 8
    x0 = x * CELL + 8
    y1 = (y + 1) * CELL - 8
    x1 = (x + 1) * CELL - 8
    cx0 = x * CELL + CELL // 2 - 4
    cx1 = x * CELL + CELL // 2 + 4
    cy0 = y * CELL + CELL // 2 - 4
    cy1 = y * CELL + CELL // 2 + 4
    if direction == Direction.UP:
        image[y0 : y0 + 6, cx0:cx1] = FACING
    elif direction == Direction.DOWN:
        image[y1 - 6 : y1, cx0:cx1] = FACING
    elif direction == Direction.LEFT:
        image[cy0:cy1, x0 : x0 + 6] = FACING
    else:
        image[cy0:cy1, x1 - 6 : x1] = FACING


def render_warehouse(env):
    warehouse = env.unwrapped
    rows, cols = warehouse.grid_size
    image = np.full((rows * CELL, cols * CELL, 3), FLOOR, dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            if warehouse.highways[y, x]:
                _fill_cell(image, x, y, AISLE, pad=1)
            y0 = y * CELL
            x0 = x * CELL
            image[y0 : y0 + 1, x0 : x0 + CELL] = GRID
            image[y0 : y0 + CELL, x0 : x0 + 1] = GRID

    for goal_x, goal_y in warehouse.goals:
        _fill_cell(image, int(goal_x), int(goal_y), GOAL, pad=4)

    requested = set(warehouse.request_queue)
    for shelf in warehouse.shelfs:
        color = REQUESTED if shelf in requested else SHELF
        _fill_cell(image, int(shelf.x), int(shelf.y), color, pad=7)

    for index, agent in enumerate(warehouse.agents):
        color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
        _fill_cell(image, int(agent.x), int(agent.y), color, pad=8)
        _paint_facing(image, int(agent.x), int(agent.y), agent.dir)
        if agent.carrying_shelf is not None:
            _fill_cell(image, int(agent.x), int(agent.y), CARRY, pad=16)

    return image


def paint(view_slot):
    actions = st.session_state.warehouse_last_actions
    rewards = st.session_state.warehouse_last_reward
    delivered = int(sum(rewards)) if rewards else 0

    with view_slot.container():
        with st.container(horizontal=True):
            st.metric("Episode", st.session_state.warehouse_episode, border=True)
            st.metric("Step", st.session_state.warehouse_step, border=True)
            st.metric(
                "Team reward",
                f"{st.session_state.warehouse_return:.0f}",
                border=True,
            )
            st.metric("Deliveries this step", delivered, border=True)

        if st.session_state.warehouse_done:
            st.info(
                "This episode reached the 500-step limit from the existing environment. "
                "The next step starts a new episode."
            )

        map_column, action_column = st.columns([3, 2], vertical_alignment="top")
        with map_column:
            st.image(
                render_warehouse(st.session_state.warehouse_env),
                caption=(
                    f"{ENV_ID} · episode {st.session_state.warehouse_episode} · "
                    f"step {st.session_state.warehouse_step}"
                ),
                channels="RGB",
                width="content",
            )
        with action_column:
            st.markdown("**Latest robot actions**")
            if actions is None:
                st.caption("No step yet. The robots are in their starting positions.")
            else:
                for row in (0, 1):
                    left, right = st.columns(2)
                    for column, index in zip((left, right), (row * 2, row * 2 + 1)):
                        with column:
                            st.metric(
                                f"Robot {index + 1}",
                                ACTION_NAMES[actions[index]],
                                f"{rewards[index]:.0f} reward",
                                border=True,
                            )


st.title("Warehouse robot navigation")
st.caption(
    "Four robots in the existing tiny warehouse. "
    "They pick up requested shelves and carry them to the delivery goals. "
    "Movement uses random legal actions until an algorithm is built."
)

st.session_state.setdefault("warehouse_seed", 42)
st.session_state.setdefault("warehouse_steps", 10)

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.number_input(
        "Seed",
        min_value=0,
        max_value=10_000,
        key="warehouse_seed",
        help="Applied when you press Reset.",
        width=160,
    )
    st.slider(
        "Steps to run",
        min_value=1,
        max_value=50,
        key="warehouse_steps",
        width=280,
    )

with st.container(horizontal=True):
    st.button(
        "Step",
        icon=":material/skip_next:",
        on_click=lambda: st.session_state.update(warehouse_command="step"),
    )
    st.button(
        "Run",
        type="primary",
        icon=":material/play_arrow:",
        on_click=lambda: st.session_state.update(warehouse_command="run"),
    )
    st.button(
        "Reset",
        icon=":material/refresh:",
        on_click=lambda: open_warehouse(st.session_state.warehouse_seed),
    )

if "warehouse_env" not in st.session_state:
    open_warehouse(st.session_state.warehouse_seed)

command = st.session_state.pop("warehouse_command", None)
view_slot = st.empty()

if command == "run":
    for _ in range(int(st.session_state.warehouse_steps)):
        step_warehouse()
        paint(view_slot)
        time.sleep(0.08)
elif command == "step":
    step_warehouse()
    paint(view_slot)
else:
    paint(view_slot)

st.caption(
    "Aisles are light gray. Purple shelves are stored, teal shelves are requested, "
    "and the dark tiles are the delivery goals. "
    "Orange, blue, green, and purple squares are the four robots. "
    "The black mark shows the direction they face, and a white center means the robot is carrying a shelf."
)
