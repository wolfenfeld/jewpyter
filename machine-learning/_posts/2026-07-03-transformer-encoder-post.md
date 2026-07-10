---
layout: post
title: "The Whole Room Is Listening — Part 3"
description: |
  Transformer Encoders: Or, How Multi-Head Attention Listens to Everything Simultaneously Without Missing the Point
image: /assets/img/DimReduction-post/cover-part1.jpg
---

*[This is Part 3. [Part 1](/machine-learning/2026-07-03-autoencoder-post/) covers autoencoders. [Part 2](/machine-learning/2026-07-03-attention-post/) covers self-attention.]*

---

At the end of Part 2, we had a problem: self-attention can look at every word from every other word, but it does not know *where* each word sits. It is like a very attentive listener who cannot remember the order in which things were said.

We also had a second problem, one Mathityahu raised on the way out.

*"One head of attention,"* he said, *"is like one person listening. They might catch the subject-verb relationship. But they might miss the pronoun reference three sentences back. What you want is the whole room listening — each person picking up something different."*

This is multi-head attention. And combined with positional encoding, it is the foundation of the transformer encoder.

---

## Problem 1: Position

A transformer has no inherent sense of order. The attention computation treats position 1 and position 10 identically — shuffle the tokens and the output changes nothing. For images this might be acceptable. For language it is fatal. *"The dog bit the man"* and *"The man bit the dog"* contain the same words. Position is the only thing separating news from miracle.

The fix: before the first attention layer, **add a position vector to each token embedding**. Each token then carries two things fused together — what it means, and where it sits.

*"Add what, exactly?"* asked Devorah.

A vector the same length as the embedding (512 numbers, say). One unique vector per position. Position 1 gets vector A, position 2 gets vector B, and so on. The model adds them component-by-component to the token embeddings. After that, two tokens with the same word but different positions have different representations going into the attention layers.

The question is: how do you design those vectors?

The simplest idea — just number them (position 1 gets the vector [1, 1, 1...], position 2 gets [2, 2, 2...]) — doesn't work. Large positions produce large numbers, the addition distorts the embeddings, and the model never generalizes to sequences longer than it saw in training.

The solution from the original 2017 paper (*Attention Is All You Need*) is to use sine and cosine waves at different frequencies. Each dimension of the position vector is assigned one wave:

<script type="math/tex; mode=display">PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)</script>

<script type="math/tex; mode=display">PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)</script>

Here *pos* is the position in the sequence, *i* is the dimension index, and *d* is the embedding size. The key idea is the denominator — 10000 raised to a power that grows with *i*. This means:

- **Low dimensions (small i)** → large denominator → fast-changing wave → nearby positions look very different from each other
- **High dimensions (large i)** → small denominator ... wait, it is the reverse: large *i* → larger exponent → larger denominator → slower wave → only distant positions differ

Dimension 0 oscillates so fast it flips sign every couple of tokens. Dimension 511 oscillates so slowly it barely moves across a sentence. Together, all 512 dimensions produce a combination of values that is unique to each position — like a clock with many hands, each ticking at a different speed. No two positions share the same combination.

### A concrete example

Take the sentence *"The bubbe made matzah ball soup again."* — seven tokens, zero-indexed. *"matzah"* sits at position 3.

To keep the numbers readable, suppose the embedding has only 8 dimensions instead of 512. The positional encoding for position 3 is computed as:

| Dimension | Formula | Value |
|---|---|---|
| 0 | sin(3 / 10000⁰) | sin(3.000) = **0.14** |
| 1 | cos(3 / 10000⁰) | cos(3.000) = **−0.99** |
| 2 | sin(3 / 10000^0.25) | sin(0.300) = **0.30** |
| 3 | cos(3 / 10000^0.25) | cos(0.300) = **0.95** |
| 4 | sin(3 / 10000^0.5) | sin(0.030) = **0.03** |
| 5 | cos(3 / 10000^0.5) | cos(0.030) = **1.00** |
| 6 | sin(3 / 10000^0.75) | sin(0.003) = **0.003** |
| 7 | cos(3 / 10000^0.75) | cos(0.003) = **1.00** |

Notice the pattern. Dimensions 0 and 1 are doing heavy oscillating — position 3 sits at a noticeable angle in a fast cycle. By dimension 4, the denominator is 100, so position 3 barely registers. By dimension 6, it is essentially zero. The fast dimensions distinguish nearby tokens; the slow ones help with tokens far apart.

Now suppose *"matzah"* has an embedding vector (its learned meaning):

```
embedding("matzah") = [ 0.80, −0.20, 0.50, 0.10, −0.30, 0.70, 0.20, −0.40 ]
```

Adding the positional encoding for position 3:

```
PE(position=3)      = [ 0.14, −0.99, 0.30, 0.95,  0.03, 1.00, 0.003, 1.00 ]
                                                                           +
input to attention  = [ 0.94, −1.19, 0.80, 1.05, −0.27, 1.70, 0.203, 0.60 ]
```

If the same word *"matzah"* appeared at position 6 instead, the positional encoding would be different — different sin/cos values, different resulting vector. Same word, different position, different input to the attention layers. That is the whole point.

```python
import torch
import numpy as np

def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: (max_len, d_model)

# Use it: add to embeddings before the attention layers
x = token_embeddings + positional_encoding(seq_len, d_model)
```

Here is what the encoding looks like across 60 positions and 64 dimensions — each row is one position's fingerprint:

<iframe src="/assets/charts/positional-encoding.html" style="width:100%;height:500px;border:none;"></iframe>

The fast oscillations on the left (low dimensions) and the slow gradients on the right (high dimensions) are visible. No two rows are the same. The model adds this matrix to the token embeddings before anything else — now every token carries both *what it is* and *where it sits*, baked into the same vector.

---

## Problem 2: One Head Is Not Enough

A single attention head learns one type of relationship. In a sentence, there are many: subject-verb agreement, pronoun reference, modifier attachment, semantic similarity. One head cannot focus on all of them simultaneously.

Multi-head attention runs *h* independent attention heads in parallel, each with its own Q, K, V projections — each free to learn different relationships. Then it concatenates the outputs and projects back to the original dimension.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.W_qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=2)

        # Split into heads: (B, n_heads, T, d_head)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention per head
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        weights = scores.softmax(dim=-1)
        out = weights @ v  # (B, n_heads, T, d_head)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out)
```

Eight heads in a 512-dimensional model: each head gets 64 dimensions. Eight different questions, asked simultaneously, about every pair of tokens. One head might track syntax. Another semantics. Another long-range dependencies. None of them know what the others found.

---

## The Full Encoder Block

One transformer encoder block combines everything: multi-head attention, a position-wise feedforward network, residual connections, and layer normalization.

<div style="overflow-x:auto;margin:2rem 0;">
<svg viewBox="0 0 300 460" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;display:block;margin:auto;font-family:sans-serif;">

  <!-- Input -->
  <rect x="90" y="10" width="120" height="36" rx="5" fill="#555"/>
  <text x="150" y="33" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Input + PE</text>

  <!-- Arrow down -->
  <line x1="150" y1="46" x2="150" y2="76" stroke="#aaa" stroke-width="1.5" marker-end="url(#a)"/>

  <!-- Multi-head attention box -->
  <rect x="60" y="76" width="180" height="44" rx="5" fill="#4fb1ba" opacity="0.85"/>
  <text x="150" y="103" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Multi-Head Attention</text>

  <!-- Skip connection 1 -->
  <line x1="60" y1="98" x2="20" y2="98" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="20" y1="98" x2="20" y2="158" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="20" y1="158" x2="60" y2="158" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#ao)"/>

  <!-- Arrow down -->
  <line x1="150" y1="120" x2="150" y2="138" stroke="#aaa" stroke-width="1.5" marker-end="url(#a)"/>

  <!-- Add & Norm 1 -->
  <rect x="60" y="138" width="180" height="36" rx="5" fill="#9b7fd4" opacity="0.75"/>
  <text x="150" y="161" text-anchor="middle" fill="white" font-size="11">Add &amp; Norm</text>

  <!-- Arrow down -->
  <line x1="150" y1="174" x2="150" y2="204" stroke="#aaa" stroke-width="1.5" marker-end="url(#a)"/>

  <!-- Feed Forward -->
  <rect x="60" y="204" width="180" height="44" rx="5" fill="#4fb1ba" opacity="0.7"/>
  <text x="150" y="226" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Feed Forward</text>
  <text x="150" y="241" text-anchor="middle" fill="white" font-size="10">Linear → ReLU → Linear</text>

  <!-- Skip connection 2 -->
  <line x1="240" y1="226" x2="276" y2="226" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="276" y1="226" x2="276" y2="290" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="276" y1="290" x2="240" y2="290" stroke="#e8a95c" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#ao)"/>

  <!-- Arrow down -->
  <line x1="150" y1="248" x2="150" y2="272" stroke="#aaa" stroke-width="1.5" marker-end="url(#a)"/>

  <!-- Add & Norm 2 -->
  <rect x="60" y="272" width="180" height="36" rx="5" fill="#9b7fd4" opacity="0.75"/>
  <text x="150" y="295" text-anchor="middle" fill="white" font-size="11">Add &amp; Norm</text>

  <!-- Arrow down -->
  <line x1="150" y1="308" x2="150" y2="336" stroke="#aaa" stroke-width="1.5" marker-end="url(#a)"/>

  <!-- Output -->
  <rect x="90" y="336" width="120" height="36" rx="5" fill="#555"/>
  <text x="150" y="359" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Output</text>

  <!-- Skip label -->
  <text x="8" y="130" fill="#e8a95c" font-size="9" font-style="italic">residual</text>
  <text x="278" y="260" fill="#e8a95c" font-size="9" font-style="italic">residual</text>

  <!-- Stack label -->
  <text x="150" y="420" text-anchor="middle" fill="#aaa" font-size="10">× N blocks stacked</text>
  <line x1="90" y1="378" x2="90" y2="405" stroke="#aaa" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="210" y1="378" x2="210" y2="405" stroke="#aaa" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="90" y1="405" x2="210" y2="405" stroke="#aaa" stroke-width="1" stroke-dasharray="3,3"/>

  <defs>
    <marker id="a" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#aaa"/>
    </marker>
    <marker id="ao" markerWidth="7" markerHeight="7" refX="2" refY="3" orient="auto">
      <path d="M7,0 L7,6 L0,3 z" fill="#e8a95c"/>
    </marker>
  </defs>
</svg>
</div>

The residual connections (shown in amber) add the block's input to its output before normalization. This means even if a block learns nothing useful, the signal still passes through unchanged. It is a safety net that makes training much more stable.

In PyTorch, using the built-in `nn.MultiheadAttention`:

```python
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention + residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward + residual
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
```

Stack N of these blocks, and you have a transformer encoder. BERT uses 12. GPT-2 uses 12. The basic block is the same; the scale differs.

---

## Putting It Together: A Full Encoder

```python
import torch
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_layers=2, d_ff=256, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('pe', self._make_pe(max_len, d_model))
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _make_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, token_ids):
        x = self.embedding(token_ids) + self.pe[:, :token_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# Quick test — no training, just shape verification
encoder = TransformerEncoder(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
tokens = torch.randint(0, 1000, (2, 20))  # batch=2, seq_len=20
output = encoder(tokens)
print(f"Input:  {tokens.shape}")   # (2, 20)
print(f"Output: {output.shape}")   # (2, 20, 128) — one 128-dim vector per token
```

Each token in the output is a 128-dimensional vector that has been informed by every other token in the sequence. This is the encoded representation. A classifier head sitting on top of this can do sentiment analysis. A cross-attention layer connected to a decoder can do translation.

---

## What BERT Actually Did

BERT (*Bidirectional Encoder Representations from Transformers*) took this architecture and trained it on a simple task: predict the missing words.

Take a sentence. Randomly mask 15% of the tokens. Train the encoder to predict what was masked, looking at the full context — left and right. Do this on the entire internet, for a very long time.

The resulting model develops representations that understand language deeply enough that with only a small amount of task-specific fine-tuning, it beats the previous state of the art on nearly every benchmark.

The encoder is not magic. It is attention plus position plus residuals plus scale. But scale, it turns out, is a large fraction of the magic.

---

## The Through-Line

We started with Devorah's shoeboxes.

An autoencoder taught us that you can compress information into a small space — and that the geometry of that compressed space is meaningful. But it treats inputs as bags of numbers, no order.

Self-attention taught us to look at everything at once — to let every token inform every other token, and to learn *what to look at* rather than being told. But a single head sees only one type of relationship, and position is invisible.

The transformer encoder brings it together: positional encoding gives every token a coordinate in the sequence; multi-head attention lets the model ask many questions simultaneously; residual connections and layer norm keep training stable; stacking the blocks builds depth.

*"So it listens,"* said Devorah, after Mathityahu had finished.

*"It listens,"* he confirmed. *"To everything. At once."*

*"Like Aunt Rivka."*

*"Exactly like Aunt Rivka. But with better memory, and more weights."*

---

*All code in this series runs on CPU. Full code: [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/).*
