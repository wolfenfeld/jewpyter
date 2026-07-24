"""Generate RL comparison chart — Hill Climb vs Q-Learning on CartPole."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# CartPole environment (no gym dependency)
# ---------------------------------------------------------------------------

class CartPole:
    """Minimal CartPole implementation matching OpenAI gym CartPole-v1."""
    GRAVITY = 9.8
    MASS_CART = 1.0
    MASS_POLE = 0.1
    POLE_HALF_LENGTH = 0.5
    FORCE_MAG = 10.0
    TAU = 0.02
    MAX_STEPS = 200

    THETA_THRESHOLD = 12 * np.pi / 180   # ~0.209 rad
    X_THRESHOLD = 2.4

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        total_mass = self.MASS_CART + self.MASS_POLE
        pole_mass_length = self.MASS_POLE * self.POLE_HALF_LENGTH
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + pole_mass_length * theta_dot ** 2 * sin_theta) / total_mass
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.POLE_HALF_LENGTH * (4 / 3 - self.MASS_POLE * cos_theta ** 2 / total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        x       += self.TAU * x_dot
        x_dot   += self.TAU * x_acc
        theta   += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = (
            abs(x) > self.X_THRESHOLD
            or abs(theta) > self.THETA_THRESHOLD
            or self.steps >= self.MAX_STEPS
        )
        reward = 1.0
        return self.state.copy(), reward, done


# ---------------------------------------------------------------------------
# Hill Climb
# ---------------------------------------------------------------------------

class HillClimbAgent:
    def __init__(self, n_features=4, n_actions=2, alpha=0.1):
        self.n_actions = n_actions
        self.weights = np.random.randn(n_features, n_actions)
        self.best_weights = self.weights.copy()
        self.best_reward = -np.inf
        self.alpha = alpha

    def get_action(self, state):
        return int(np.argmax(np.dot(state, self.weights)))

    def update(self, total_reward):
        if total_reward >= self.best_reward:
            self.best_reward = total_reward
            self.best_weights = self.weights.copy()
            self.alpha = max(self.alpha / 1.5, 1e-2)
        else:
            self.alpha = min(self.alpha * 2, 2.0)
        self.weights = self.best_weights + self.alpha * np.random.randn(*self.weights.shape)


def run_hill_climb(episodes=300):
    env = CartPole()
    agent = HillClimbAgent()
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
# Q-Learning
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, n_actions=2, n_bins=10, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.q_table = {}

    def _discretize(self, state):
        bins = np.linspace(-3, 3, self.n_bins)
        return tuple(np.digitize(np.clip(s, -3, 3), bins) for s in state)

    def _q(self, state):
        key = self._discretize(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._q(state)))

    def update(self, state, action, reward, next_state, done):
        q = self._q(state)
        next_q = self._q(next_state) if not done else np.zeros(self.n_actions)
        q[action] += self.alpha * (reward + self.gamma * np.max(next_q) - q[action])
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def run_q_learning(episodes=300):
    env = CartPole()
    agent = QLearningAgent()
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
        rewards.append(total)

    return np.array(rewards)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save(fig, name):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    style = "<style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}</style>"
    html = html.replace("</head>", style + "</head>", 1)
    with open(f"assets/charts/{name}.html", "w") as f:
        f.write(html)
    print(f"Saved {name}.html")


def smooth(arr, window=10):
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def chart_rl_comparison():
    print("Chart — Hill Climb vs Q-Learning (CartPole)")
    N_RUNS = 50
    EPISODES = 300

    hc_all = np.zeros((N_RUNS, EPISODES))
    ql_all = np.zeros((N_RUNS, EPISODES))

    for i in range(N_RUNS):
        np.random.seed(i)
        hc_all[i] = run_hill_climb(EPISODES)
        np.random.seed(i)
        ql_all[i] = run_q_learning(EPISODES)

    hc_mean = hc_all.mean(axis=0)
    hc_std  = hc_all.std(axis=0)
    ql_mean = ql_all.mean(axis=0)
    ql_std  = ql_all.std(axis=0)

    episodes = np.arange(1, EPISODES + 1)

    fig = go.Figure()

    # Hill Climb band + line
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

    # Q-Learning band + line
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
        title="CartPole — Hill Climb vs Q-Learning (50 runs)",
        xaxis_title="Episode",
        yaxis_title="Reward",
        yaxis_range=[0, 210],
        legend=dict(x=0.02, y=0.98),
    )
    save(fig, "rl-comparison")


if __name__ == "__main__":
    chart_rl_comparison()
