"""Generate FrozenLake comparison chart — Hill Climb vs Q-Learning."""

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# FrozenLake environment (no gym dependency)
# Standard 4x4 map: S=start, F=frozen, H=hole, G=goal
# S F F F
# F H F H
# F F F H
# H F F G
# ---------------------------------------------------------------------------

MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]
NROW, NCOL = 4, 4
HOLES = {r * NCOL + c for r, row in enumerate(MAP) for c, ch in enumerate(row) if ch == "H"}
GOAL  = NROW * NCOL - 1  # state 15
START = 0

# Actions: 0=left, 1=down, 2=right, 3=up
DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


class FrozenLake:
    MAX_STEPS = 100

    def reset(self):
        self.state = START
        self.steps = 0
        return self.state

    def step(self, action):
        r, c = divmod(self.state, NCOL)
        dr, dc = DELTAS[action]
        nr = max(0, min(NROW - 1, r + dr))
        nc = max(0, min(NCOL - 1, c + dc))
        self.state = nr * NCOL + nc
        self.steps += 1

        if self.state in HOLES:
            return self.state, 0.0, True
        if self.state == GOAL:
            return self.state, 1.0, True
        if self.steps >= self.MAX_STEPS:
            return self.state, 0.0, True
        return self.state, 0.0, False


# ---------------------------------------------------------------------------
# Hill Climb — one-hot state encoding, linear policy
# ---------------------------------------------------------------------------

N_STATES  = NROW * NCOL  # 16
N_ACTIONS = 4

def one_hot(state):
    v = np.zeros(N_STATES)
    v[state] = 1.0
    return v


class HillClimbAgentFL:
    def __init__(self, alpha=0.1):
        self.weights = np.random.randn(N_STATES, N_ACTIONS)
        self.best_weights = self.weights.copy()
        self.best_reward = -np.inf
        self.alpha = alpha

    def get_action(self, state):
        return int(np.argmax(one_hot(state) @ self.weights))

    def update(self, total_reward):
        if total_reward >= self.best_reward:
            self.best_reward = total_reward
            self.best_weights = self.weights.copy()
            self.alpha = max(self.alpha / 1.5, 1e-3)
        else:
            self.alpha = min(self.alpha * 2, 2.0)
        self.weights = self.best_weights + self.alpha * np.random.randn(N_STATES, N_ACTIONS)


def run_hill_climb(episodes=1000):
    env = FrozenLake()
    agent = HillClimbAgentFL()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        while True:
            action = agent.get_action(state)
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        agent.update(total)
        rewards.append(total)
    return np.array(rewards)


# ---------------------------------------------------------------------------
# Q-Learning — exact tabular (16 states × 4 actions)
# ---------------------------------------------------------------------------

class QLearningAgentFL:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.q = np.zeros((N_STATES, N_ACTIONS))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        next_q = 0.0 if done else np.max(self.q[next_state])
        td_error = reward + self.gamma * next_q - self.q[state, action]
        self.q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def run_q_learning(episodes=1000):
    env = FrozenLake()
    agent = QLearningAgentFL()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
            if done:
                break
        agent.decay_epsilon()
        rewards.append(total)
    return np.array(rewards)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def rolling_mean(arr, window=20):
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def save(fig, name):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    style = "<style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}</style>"
    html = html.replace("</head>", style + "</head>", 1)
    with open(f"assets/charts/{name}.html", "w") as f:
        f.write(html)
    print(f"Saved {name}.html")


def chart_frozenlake_comparison():
    print("Chart — FrozenLake: Hill Climb vs Q-Learning")
    N_RUNS   = 50
    EPISODES = 1000
    WINDOW   = 30

    hc_all = np.zeros((N_RUNS, EPISODES - WINDOW + 1))
    ql_all = np.zeros((N_RUNS, EPISODES - WINDOW + 1))

    for i in range(N_RUNS):
        np.random.seed(i)
        hc_all[i] = rolling_mean(run_hill_climb(EPISODES), WINDOW)
        np.random.seed(i)
        ql_all[i] = rolling_mean(run_q_learning(EPISODES), WINDOW)

    hc_mean = hc_all.mean(axis=0)
    hc_std  = hc_all.std(axis=0)
    ql_mean = ql_all.mean(axis=0)
    ql_std  = ql_all.std(axis=0)

    episodes = np.arange(WINDOW, EPISODES + 1)


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([episodes, episodes[::-1]]),
        y=np.concatenate([hc_mean + hc_std, (hc_mean - hc_std)[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=hc_mean, name="Hill Climb",
        line=dict(color="#636EFA", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([episodes, episodes[::-1]]),
        y=np.concatenate([ql_mean + ql_std, (ql_mean - ql_std)[::-1]]),
        fill="toself", fillcolor="rgba(239,85,59,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=ql_mean, name="Q-Learning",
        line=dict(color="#EF553B", width=2),
    ))

    fig.update_layout(
        title="FrozenLake — Hill Climb vs Q-Learning (50 runs, 30-episode rolling success rate)",
        xaxis_title="Episode",
        yaxis_title="Success rate (rolling avg)",
        yaxis_range=[0, 1],
        legend=dict(x=0.02, y=0.98),
    )
    save(fig, "frozenlake-comparison")


if __name__ == "__main__":
    chart_frozenlake_comparison()
