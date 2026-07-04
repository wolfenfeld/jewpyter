---
layout: post
title: "The Forgetful Scribe — Part 1"
description: |
  Autoencoders: Or, How to Compress Everything You Know Into Two Numbers and Still Recognize Your Cousin
image: /assets/img/DimReduction-post/cover-part1.jpg
---

Mathityahu had a cousin — Devorah — who kept every photograph the family had ever taken. Shoeboxes. Closets. An entire drawer dedicated solely to blurry Bar Mitzvah photos from 1994. Her apartment was one fire hazard away from a documentary.

*"I need to store these somewhere,"* she told Mathityahu. *"But there are thousands of them."*

*"How much do you actually need to remember?"* he asked.

She thought about it. *"Enough to recognize who's who. Enough to reconstruct the memory."*

*"Good,"* said Mathityahu. *"Then we don't need to store the whole thing. We need to store the essence."*

This is the autoencoder. It does not keep everything. It learns what to keep.

---

## The Idea: Compress, Then Reconstruct

An autoencoder is a neural network with a bottleneck in the middle. The left half — the **encoder** — takes your input and squeezes it into a small representation. The right half — the **decoder** — tries to reconstruct the original from that small representation.

The network is trained on its own mistakes: if the reconstruction is bad, it learns to compress better.

<div style="overflow-x:auto;margin:2rem 0;">
<svg viewBox="0 0 780 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:780px;display:block;margin:auto;font-family:sans-serif;">
  <!-- Input -->
  <rect x="10" y="20" width="60" height="140" rx="6" fill="#4fb1ba" opacity="0.85"/>
  <text x="40" y="96" text-anchor="middle" fill="white" font-size="11" font-weight="bold">784</text>
  <text x="40" y="172" text-anchor="middle" fill="#555" font-size="10">Input</text>

  <!-- Arrow -->
  <line x1="70" y1="90" x2="110" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Layer 256 -->
  <rect x="110" y="35" width="50" height="110" rx="6" fill="#4fb1ba" opacity="0.65"/>
  <text x="135" y="96" text-anchor="middle" fill="white" font-size="11">256</text>

  <line x1="160" y1="90" x2="200" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Layer 64 -->
  <rect x="200" y="55" width="50" height="70" rx="6" fill="#4fb1ba" opacity="0.5"/>
  <text x="225" y="96" text-anchor="middle" fill="white" font-size="11">64</text>

  <line x1="250" y1="90" x2="300" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Bottleneck -->
  <rect x="300" y="72" width="50" height="36" rx="6" fill="#e8a95c"/>
  <text x="325" y="94" text-anchor="middle" fill="white" font-size="11" font-weight="bold">2</text>
  <text x="325" y="125" text-anchor="middle" fill="#e8a95c" font-size="10" font-weight="bold">bottleneck</text>

  <line x1="350" y1="90" x2="400" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Layer 64 decoder -->
  <rect x="400" y="55" width="50" height="70" rx="6" fill="#9b7fd4" opacity="0.5"/>
  <text x="425" y="96" text-anchor="middle" fill="white" font-size="11">64</text>

  <line x1="450" y1="90" x2="490" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Layer 256 decoder -->
  <rect x="490" y="35" width="50" height="110" rx="6" fill="#9b7fd4" opacity="0.65"/>
  <text x="515" y="96" text-anchor="middle" fill="white" font-size="11">256</text>

  <line x1="540" y1="90" x2="580" y2="90" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Output -->
  <rect x="580" y="20" width="60" height="140" rx="6" fill="#9b7fd4" opacity="0.85"/>
  <text x="610" y="96" text-anchor="middle" fill="white" font-size="11" font-weight="bold">784</text>
  <text x="610" y="172" text-anchor="middle" fill="#555" font-size="10">Output</text>

  <!-- Labels -->
  <text x="200" y="16" text-anchor="middle" fill="#4fb1ba" font-size="11" font-weight="bold">ENCODER</text>
  <text x="490" y="16" text-anchor="middle" fill="#9b7fd4" font-size="11" font-weight="bold">DECODER</text>

  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#aaa"/>
    </marker>
  </defs>
</svg>
</div>

The bottleneck forces the network to prioritize. It cannot copy everything through — there are only two numbers to work with. So it learns to encode what matters and throw away what doesn't.

This is, Mathityahu explained to Devorah, how the brain remembers a face. Not every pixel. The structure.

---

## Why Multiple Layers?

*"Why not go straight from 784 to 2?"* asked Devorah. *"Why the stops in between?"*

A fair question. You could build a single-layer autoencoder that goes directly: 784 → 2 → 784. It would technically work. It would also be terrible.

The reason is that the relationship between a raw pixel grid and a meaningful concept — "this is a 3," "this is a 7" — is not linear. It is built up in stages. Edges first, then curves, then strokes, then digits. Each layer in the encoder learns to detect a slightly higher-level abstraction than the one before it.

- **Layer 784 → 256**: Detects local patterns — edges, corners, small strokes
- **Layer 256 → 64**: Combines those patterns — loops, lines, junctions
- **Layer 64 → 2**: Captures global structure — what kind of digit this is

A direct jump from 784 to 2 forces the network to learn all of this in a single linear transformation, which it cannot. The intermediate layers give it room to build concepts gradually.

The decoder mirrors this in reverse — from a global concept, it reconstructs strokes, then pixels.

*"It's like translating,"* said Devorah. *"Word by word doesn't work. You need to understand the sentence first."*

*"Exactly,"* said Mathityahu.

---

## How Do You Choose Layer Sizes?

There is no formula. There are heuristics.

The general principle is a **gradual funnel**: each encoder layer should be meaningfully smaller than the previous one, but not so small that it drops information too abruptly. A common rule of thumb is to halve (or quarter) the size at each step.

For 784 → 2, a reasonable progression might be:

| Approach | Layers |
|---|---|
| Aggressive | 784 → 64 → 2 |
| Moderate (ours) | 784 → 256 → 64 → 2 |
| Conservative | 784 → 512 → 128 → 32 → 2 |

More layers give the network more capacity to learn complex representations — but also more parameters to train, longer training time, and higher risk of overfitting. For MNIST, the moderate approach is plenty.

The bottleneck size is the most important choice. It controls the information budget:

- **Too small** (e.g., 2 for complex data): blurry, lossy reconstructions. Forces the network to throw away too much.
- **Too large** (e.g., 256 for simple data): the network memorizes instead of compressing. The latent space is a mess.
- **Just right**: depends on the complexity of what you want to represent. For MNIST digits, 2 is aggressive but works. For faces, you'd want 64–256.

In practice, you try a few values, look at the reconstructions, and decide whether the blurriness is acceptable.

---

## The Loss Function

The loss function is the measure of wrongness. It is what the network is trying to minimize.

For autoencoders, the natural choice is **reconstruction loss** — how different is the output from the input?

The most common version is **Mean Squared Error (MSE)**:

<script type="math/tex; mode=display">\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2</script>

For each pixel, square the difference between the original value and the reconstructed value, then average across all pixels and all examples in the batch. Large errors are penalized more than small ones (because of the squaring).

In PyTorch:

```python
criterion = nn.MSELoss()
loss = criterion(reconstruction, original)
```

An alternative for images where pixel values are between 0 and 1 is **Binary Cross-Entropy (BCE)**:

<script type="math/tex; mode=display">\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]</script>

BCE treats each pixel as a Bernoulli variable — "is this pixel on or off?" — and penalizes confident wrong predictions harshly. It often produces sharper reconstructions than MSE but requires a Sigmoid activation on the decoder's final layer (which we have).

For simplicity, we use MSE. The reconstructions are blurrier than BCE would give, but the latent space structure is just as informative.

---

## The Dataset and Training

We will use MNIST — sixty thousand handwritten digits, each 28×28 pixels. Every image is 784 numbers. We will compress them into 2.

Why 2? Because 2 is the smallest number where you can still make a picture.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64, 256),         nn.ReLU(),
            nn.Linear(256, 784),        nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
```

The training loop is simple. Show the network an image. Get the reconstruction. Measure how wrong it is. Make it less wrong.

```python
transform = transforms.ToTensor()
data = datasets.MNIST('.', train=True, download=True, transform=transform)
loader = DataLoader(data, batch_size=256, shuffle=True)

model = Autoencoder(latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epoch_losses = []
for epoch in range(20):
    total_loss = 0
    for images, _ in loader:
        recon, z = model(images)
        loss = criterion(recon, images.view(-1, 784))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    epoch_losses.append(avg)
    print(f"Epoch {epoch+1}: loss = {avg:.4f}")
```

Twenty epochs. Runs on a laptop in about two minutes. No GPU required.

---

## Watching the Network Learn

Each epoch, the network sees the entire training set once and adjusts its weights to reduce the loss. Here is what that looks like:

<iframe src="/assets/charts/autoencoder-training-loss.html" style="width:100%;height:480px;border:none;"></iframe>

The loss drops steeply in the first few epochs — the network is learning the obvious structure quickly. Then it flattens as the easy gains are exhausted and only the harder details remain.

When do you stop? You watch the curve. When it flattens and the reconstructions look acceptable, you stop. There is no rule that says "twenty epochs." Twenty was a judgment call based on watching this particular curve. In practice you would also track the loss on a held-out **validation set** — if the training loss keeps dropping but the validation loss starts rising, you are overfitting and should stop.

---

## What Did It Learn?

After training, we take the test set — digits the model has never seen — and encode each one into its two latent numbers. Then we plot them:

<iframe src="/assets/charts/autoencoder-latent-space.html" style="width:100%;height:500px;border:none;"></iframe>

Nobody told the autoencoder that there are ten different digits. Nobody told it what a "3" is or how it differs from an "8." It figured this out by learning to compress and reconstruct — and in doing so, it discovered that digits cluster.

The geometry of the latent space is what the network thinks matters. Similar shapes land near each other. Different ones land far apart.

Now look at what it actually remembers:

<iframe src="/assets/charts/autoencoder-reconstructions.html" style="width:100%;height:340px;border:none;"></iframe>

The top row is the original. The bottom row is the reconstruction from just two numbers. It is blurry — two numbers cannot hold everything. But you can read it. The structure survived.

*"It forgets the handwriting,"* said Devorah, looking at the outputs. *"But it remembers the digit."*

*"That is the point,"* said Mathityahu.

---

## What Is the Latent Space, Exactly?

The latent space is the set of all points the encoder can produce — all possible compressions of all possible inputs.

In our case it is two-dimensional, so it is literally a plane. Every digit from the test set maps to a point on that plane. The position is determined by the encoder's learned weights — it is not random, and it is not arbitrary.

What makes the latent space interesting is its **geometry**. Because the network was trained to reconstruct inputs from their latent codes, similar inputs must map to similar codes — otherwise a small perturbation in the latent space would produce a wildly different reconstruction. The training pressure pushes related things together and unrelated things apart.

This is why digits cluster without being told to. A "3" and another "3" produce similar pixel patterns, so the encoder has to give them similar latent codes in order to reconstruct both correctly from a single decoder. A "3" and a "7" produce very different patterns, so they end up far apart.

Notice in the plot that some clusters overlap — "4" and "9" sit close together, as do "3" and "8". This makes sense: they share visual structure (loops, curved strokes). The two-dimensional space does not have enough room to separate everything cleanly, which is why 2D is good for visualization but a real application would use a larger latent dimension.

---

## What Is the Latent Space Used For?

*"So we have this map,"* said Devorah. *"What do we actually do with it?"*

Several things.

**1. Compression and search.** Store the latent code instead of the original. A 2-number code instead of 784. To find similar images, compare codes — it is much faster than comparing full images pixel by pixel.

**2. Anomaly detection.** Train the autoencoder on normal data. Then, for a new input, measure the reconstruction error. If the input is similar to what the network has seen, it will reconstruct it well. If it is unusual — a corrupted image, a fraud transaction, a machine behaving strangely — the reconstruction will be poor and the error will be high. High error = anomaly.

**3. Interpolation.** Take the latent code of a "3" and the latent code of a "8". Average them. Decode the result. You get something that looks like a digit halfway between the two — a smoothly blended hallucination. This works because the latent space is continuous: nearby codes decode to similar-looking images.

**4. Generation.** Sample a random point from the latent space. Decode it. You get a new image that looks like something the network has seen, but that never existed in the training data. This is the seed of the idea behind generative models — though a plain autoencoder is not very good at this. The latent space is not guaranteed to be smooth, so many random points decode to noise. The **Variational Autoencoder (VAE)** fixes this by explicitly shaping the latent space into a smooth distribution — but that is a story for another day.
**5. Transfer learning.** The encoder, trained on one task, can be reused as a feature extractor for another. Because reconstruction forces the network to understand structure — what makes a 7 look different from a 1 — the latent code carries meaning that no label was needed to produce. Freeze the encoder, attach a small classifier on top, train it on a handful of labeled examples, and the encoder's representations do the heavy lifting. This is the same principle behind BERT and GPT: pretrain cheaply on unlabeled data, fine-tune efficiently on expensive labeled data.
---

## The Limitation

The autoencoder solves one problem very well: compress an image, reconstruct it later.

But Devorah had a follow-up question. *"What if I don't want images? What if I want to understand text? A sentence?"*

Mathityahu considered this. *"A sentence is not like an image. An image doesn't care about order — you can rotate it. A sentence does care. 'The dog bit the man' and 'The man bit the dog' use the same words. One is news. One is a miracle."*

The autoencoder has no concept of order. It sees inputs as a flat bag of numbers. Rearrange them and it doesn't notice.

For sequences — text, speech, events in time — we need something that reads. Something that understands that what came before changes the meaning of what comes next.

*"We need the other guys,"* said Mathityahu.

*[Continue to [Part 2 — The Scribe Reads the Room](/machine-learning/2026-07-03-attention-post/)]*

---

*All code for this post runs on CPU. Dataset downloads automatically on first run (~11MB). Full code: [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/).*
