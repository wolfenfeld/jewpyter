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

## The Dataset: MNIST Handwritten Digits

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

for epoch in range(20):
    for images, _ in loader:
        recon, z = model(images)
        loss = criterion(recon, images.view(-1, 784))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
```

Twenty epochs. Runs on a laptop in about two minutes. No GPU required.

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

## The Limitation

The autoencoder solves one problem very well: compress an image, reconstruct it later.

But Devorah had a follow-up question. *"What if I don't want images? What if I want to understand text? A sentence?"*

Mathityahu considered this. *"A sentence is not like an image. An image doesn't care about order — you can rotate it. A sentence does care. 'The dog bit the man' and 'The man bit the dog' use the same words. One is news. One is a miracle."*

The autoencoder has no concept of order. It sees inputs as a flat bag of numbers. Rearrange them and it doesn't notice.

For sequences — text, speech, events in time — we need something that reads. Something that understands that what came before changes the meaning of what comes next.

*"We need the other guys,"* said Mathityahu.

*[Continue to [Part 2 — The Scribe Reads the Room](/data-science/2026-07-12-attention-post/)]*

---

*All code for this post runs on CPU. Dataset downloads automatically on first run (~11MB). Full code: [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/).*
