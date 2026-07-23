"""Generate MAB simulation chart — UCB vs Thompson Sampling cumulative regret."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import plotly.graph_objects as go


def save(fig, name):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    style = "<style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}</style>"
    html = html.replace("</head>", style + "</head>", 1)
    with open(f"assets/charts/{name}.html", "w") as f:
        f.write(html)
    print(f"Saved {name}.html")


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        return 1 if np.random.random() < self.p else 0


class UCB:
    def __init__(self, n_arms, beta=1.0):
        self.n_arms = n_arms
        self.beta = beta
        self.counts = np.zeros(n_arms)
        self.mu_hat = np.zeros(n_arms)

    def select_arm(self, t):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.mu_hat + self.beta * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mu_hat[arm] = ((n - 1) * self.mu_hat[arm] + reward) / n


class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self):
        samples = np.random.beta(self.successes, self.failures)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.successes[arm] += reward
        self.failures[arm] += 1 - reward


def run_simulation(n_arms=5, n_rounds=1000, n_runs=500, seed=42):
    np.random.seed(seed)

    ucb_regrets = np.zeros((n_runs, n_rounds))
    ts_regrets = np.zeros((n_runs, n_rounds))

    for run in range(n_runs):
        means = np.random.uniform(0.1, 0.9, n_arms)
        best_mean = np.max(means)
        arms = [BernoulliArm(p) for p in means]

        ucb = UCB(n_arms)
        ts = ThompsonSampling(n_arms)

        ucb_cumulative = 0
        ts_cumulative = 0

        for t in range(1, n_rounds + 1):
            arm_ucb = ucb.select_arm(t)
            reward_ucb = arms[arm_ucb].draw()
            ucb.update(arm_ucb, reward_ucb)
            ucb_cumulative += best_mean - means[arm_ucb]
            ucb_regrets[run, t - 1] = ucb_cumulative

            arm_ts = ts.select_arm()
            reward_ts = arms[arm_ts].draw()
            ts.update(arm_ts, reward_ts)
            ts_cumulative += best_mean - means[arm_ts]
            ts_regrets[run, t - 1] = ts_cumulative

    return ucb_regrets, ts_regrets


def chart_mab_regret():
    print("Chart — MAB cumulative regret")
    ucb_regrets, ts_regrets = run_simulation()

    rounds = np.arange(1, ucb_regrets.shape[1] + 1)

    ucb_mean = ucb_regrets.mean(axis=0)
    ucb_std = ucb_regrets.std(axis=0)
    ts_mean = ts_regrets.mean(axis=0)
    ts_std = ts_regrets.std(axis=0)

    fig = go.Figure()

    # UCB band
    fig.add_trace(go.Scatter(
        x=np.concatenate([rounds, rounds[::-1]]),
        y=np.concatenate([ucb_mean + ucb_std, (ucb_mean - ucb_std)[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=ucb_mean,
        mode="lines", name="UCB",
        line=dict(color="#636EFA", width=2),
    ))

    # Thompson Sampling band
    fig.add_trace(go.Scatter(
        x=np.concatenate([rounds, rounds[::-1]]),
        y=np.concatenate([ts_mean + ts_std, (ts_mean - ts_std)[::-1]]),
        fill="toself", fillcolor="rgba(239,85,59,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=ts_mean,
        mode="lines", name="Thompson Sampling",
        line=dict(color="#EF553B", width=2),
    ))

    fig.update_layout(
        title="Cumulative Regret — UCB vs Thompson Sampling (500 runs, 5 arms)",
        xaxis_title="Round",
        yaxis_title="Cumulative Regret",
        legend=dict(x=0.02, y=0.98),
    )
    save(fig, "mab-regret")


if __name__ == "__main__":
    chart_mab_regret()
