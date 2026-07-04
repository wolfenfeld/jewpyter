"""
Generates Plotly charts for the Transformer Encoder series posts.

Outputs:
  - assets/charts/autoencoder-latent-space.html
  - assets/charts/autoencoder-reconstructions.html
  - assets/charts/attention-heatmap.html
  - assets/charts/positional-encoding.html

Requirements:
    pip install plotly torch torchvision numpy

Usage:
    python scripts/generate_transformer_charts.py
"""

import os
import ssl
import urllib.request
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Fix SSL cert issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
from plotly.subplots import make_subplots

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "../assets/charts")


def save(fig, name, height=480):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    html_path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.update_layout(height=height)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    with open(html_path, "r") as f:
        html = f.read()
    html = html.replace(
        "<head>",
        "<head><style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}</style>"
    )
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  ✓  {html_path}")


def chart_autoencoder_latent_space():
    """Train a 2D autoencoder on MNIST and plot the latent space."""
    print("Chart — Autoencoder latent space  [training ~2 min on CPU]")

    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=256, shuffle=True)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(),
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 784), nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epoch_losses = []
    for epoch in range(20):
        total_loss = 0
        for images, _ in loader:
            recon, _ = model(images)
            loss = criterion(recon, images.view(-1, 784))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        epoch_losses.append(avg)
        print(f"    epoch {epoch + 1}/20  loss={avg:.4f}")

    # Encode 5000 test samples
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=5000, shuffle=True)
    images, labels = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        _, z = model(images)

    z = z.numpy()
    labels = labels.numpy()

    fig = px.scatter(
        x=z[:, 0], y=z[:, 1],
        color=labels.astype(str),
        labels={"x": "Latent Dimension 1", "y": "Latent Dimension 2", "color": "Digit"},
        title="MNIST Digits — 784 Dimensions Squeezed into 2",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(legend_title_text="Digit")
    save(fig, "autoencoder-latent-space")

    return model, epoch_losses


def chart_autoencoder_training_loss(epoch_losses):
    """Plot training loss curve over epochs."""
    print("Chart — Training loss curve")

    epochs = list(range(1, len(epoch_losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=epoch_losses,
        mode="lines+markers",
        line=dict(color="#4fb1ba", width=2.5),
        marker=dict(size=7, color="#4fb1ba"),
        name="Training Loss",
    ))
    fig.update_layout(
        title="Training Loss — The Network Gets Less Wrong Every Epoch",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        showlegend=False,
    )
    save(fig, "autoencoder-training-loss")


def chart_autoencoder_reconstructions(model):
    """Show a grid of original vs reconstructed digits."""
    print("Chart — Autoencoder reconstructions")

    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = DataLoader(test_data, batch_size=10, shuffle=True)
    images, labels = next(iter(loader))

    model.eval()
    with torch.no_grad():
        recon, _ = model(images)

    originals = images.squeeze().numpy()
    reconstructed = recon.view(-1, 28, 28).numpy()

    titles = [f"<b>{l.item()}</b>" for l in labels] + [""] * 10
    fig = make_subplots(
        rows=2, cols=10,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.01,
    )

    for i in range(10):
        fig.add_trace(go.Heatmap(
            z=originals[i], colorscale="gray_r",
            showscale=False, zmin=0, zmax=1,
        ), row=1, col=i + 1)
        fig.add_trace(go.Heatmap(
            z=reconstructed[i], colorscale="gray_r",
            showscale=False, zmin=0, zmax=1,
        ), row=2, col=i + 1)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(
        title="Original (top) vs Reconstructed (bottom) — What Survives the Bottleneck",
        margin=dict(t=60, b=20),
    )
    save(fig, "autoencoder-reconstructions", height=320)


def chart_attention_heatmap():
    """Illustrative self-attention weight heatmap."""
    print("Chart — Self-attention heatmap")

    tokens = ["The", "bubbe", "made", "matzah", "ball", "soup", "again"]
    n = len(tokens)

    # Construct a plausible attention pattern
    np.random.seed(17)
    weights = np.random.dirichlet(np.ones(n) * 0.3, size=n)

    # Add structure: self-attention diagonal + linguistic patterns
    for i in range(n):
        weights[i, i] += 0.25
    weights[3, 1] += 0.35   # matzah → bubbe
    weights[4, 3] += 0.45   # ball → matzah
    weights[5, 2] += 0.30   # soup → made
    weights[2, 1] += 0.25   # made → bubbe
    weights[6, 2] += 0.20   # again → made

    # Renormalize rows
    weights = weights / weights.sum(axis=1, keepdims=True)

    fig = go.Figure(go.Heatmap(
        z=weights,
        x=tokens, y=tokens,
        colorscale="Blues",
        zmin=0, zmax=weights.max(),
        text=np.round(weights, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title="Self-Attention — What Each Word Is Looking At",
        xaxis_title="Keys (attended to)",
        yaxis_title="Queries (attending)",
        yaxis_autorange="reversed",
    )
    save(fig, "attention-heatmap")


def chart_positional_encoding():
    """Sinusoidal positional encoding heatmap."""
    print("Chart — Positional encoding")

    max_pos = 60
    d_model = 64

    pe = np.zeros((max_pos, d_model))
    position = np.arange(max_pos)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    fig = go.Figure(go.Heatmap(
        z=pe,
        colorscale="RdBu",
        zmin=-1, zmax=1,
    ))
    fig.update_layout(
        title="Positional Encoding — Each Position Gets a Unique Fingerprint",
        xaxis_title="Embedding Dimension",
        yaxis_title="Position in Sequence",
    )
    save(fig, "positional-encoding")


def chart_autoencoder_transfer(model):
    """Compare transfer learning (32D encoder) vs raw pixels at different labeled-data sizes."""
    print("Chart — Transfer learning vs raw pixels  [training 32D encoder ~2 min]")

    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Train a separate 32D autoencoder — more useful for transfer than 2D
    class Autoencoder32(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 64),  nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),  nn.ReLU(),
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 784), nn.Sigmoid(),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model32 = Autoencoder32()
    opt = torch.optim.Adam(model32.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    loader_full = DataLoader(train_data, batch_size=256, shuffle=True)
    for epoch in range(20):
        for imgs, _ in loader_full:
            recon, _ = model32(imgs)
            loss = crit(recon, imgs.view(-1, 784))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"    32D epoch {epoch+1}/20  loss={loss.item():.4f}")

    model32.eval()

    def encode_all(ae, dataset):
        loader = DataLoader(dataset, batch_size=1000)
        zs, ys = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                zs.append(ae.encoder(imgs).numpy())
                ys.append(labels.numpy())
        import numpy as np
        return np.concatenate(zs), np.concatenate(ys)

    import numpy as np
    Z_train, y_train = encode_all(model32, train_data)
    Z_test,  y_test  = encode_all(model32, test_data)

    # Raw pixels
    X_train = np.array([img.numpy().flatten() for img, _ in train_data])
    X_test  = np.array([img.numpy().flatten() for img, _ in test_data])

    label_counts = [50, 100, 200, 500, 1000, 2000, 5000]
    acc_latent, acc_pixels = [], []

    for n in label_counts:
        # Sample n balanced examples (n // 10 per class)
        idx = []
        for cls in range(10):
            cls_idx = np.where(y_train == cls)[0][:n // 10]
            idx.extend(cls_idx)
        idx = np.array(idx)

        # Latent classifier
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(scaler.fit_transform(Z_train[idx]), y_train[idx])
        acc_latent.append(clf.score(scaler.transform(Z_test), y_test) * 100)

        # Raw pixel classifier
        scaler2 = StandardScaler()
        clf2 = LogisticRegression(max_iter=500, C=0.1)
        clf2.fit(scaler2.fit_transform(X_train[idx]), y_train[idx])
        acc_pixels.append(clf2.score(scaler2.transform(X_test), y_test) * 100)

        print(f"    n={n:5d}  latent={acc_latent[-1]:.1f}%  pixels={acc_pixels[-1]:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=label_counts, y=acc_latent,
        mode="lines+markers", name="Latent code (2D) + linear classifier",
        line=dict(color="#4fb1ba", width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=label_counts, y=acc_pixels,
        mode="lines+markers", name="Raw pixels (784D) + logistic regression",
        line=dict(color="#e8a95c", width=2.5, dash="dash"),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Transfer Learning — Latent Codes Beat Raw Pixels When Labels Are Scarce",
        xaxis_title="Number of Labeled Training Examples",
        yaxis_title="Test Accuracy (%)",
        xaxis=dict(type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_range=[40, 100],
    )
    save(fig, "autoencoder-transfer")


if __name__ == "__main__":
    print("\nGenerating transformer series charts...\n")
    model, losses = chart_autoencoder_latent_space()
    chart_autoencoder_training_loss(losses)
    chart_autoencoder_reconstructions(model)
    chart_autoencoder_transfer(model)
    chart_attention_heatmap()
    chart_positional_encoding()
    print("\nDone.\n  Charts → assets/charts/")
