---
layout: post
title: "The Scribe Reads the Room — Part 2"
description: |
  Self-Attention: Or, How to Listen to Everyone at Once Without Going Completely Meshuggeneh
image: /assets/img/DimReduction-post/cover-part1.jpg
---

*[This is Part 2. [Part 1 — The Forgetful Scribe](/machine-learning/2026-07-03-autoencoder-post/) covers autoencoders.]*

---

Mathityahu's aunt Rivka was a very good listener. At any family gathering, she could track fourteen conversations simultaneously — who said what to whom, who owed whom an apology, which cousin still hadn't called his mother. She did not need to wait until the end of dinner to understand the beginning of dinner. Everything informed everything else, in real time.

*"This,"* said Mathityahu, *"is what we need the machine to do."*

The autoencoder, as we saw in Part 1, forgets the order. It compresses everything into a single small representation and reconstructs from that. For images, this works. For language — for anything where sequence matters — it doesn't. A sentence is not a bag of words. It is a structure.

The field tried to solve this with sequential models. Read left to right, carry the memory forward. But the memory fades. By the time you reach the end of a long sentence, the beginning is a blur — like trying to remember the first course at a Pesach seder after you've already reached the afikomen.

The solution was something else entirely. Instead of reading left to right and forgetting, look at everything at once — and learn *what to look at*.

This is **self-attention**.

---

## The Bottleneck Problem

Before we solve it, let us be precise about what is broken.

Imagine encoding the sentence *"The matzah, which the bubbe made from scratch and which took her three hours and two arguments with Uncle Shimon, was excellent"* into a single fixed-size vector.

By the time you encode *"was excellent"*, the model must somehow still remember *"matzah"* — the subject — despite everything that came in between. In practice, it doesn't. It remembers the most recent things well, and the earlier things poorly.

This is the bottleneck. Not size this time — time. The information has to travel through a long chain, and it degrades.

---

## The Fix: Look at Everything at Once

Self-attention abandons the sequential constraint entirely. Instead of passing information down a chain, every token looks at every other token directly and decides for itself what to pay attention to.

Take the sentence: *"The bubbe made matzah ball soup again."*

When the model is processing the word *"ball,"* it needs context. *"Ball"* alone could mean anything — basketball, formal dance, a good time. What resolves the ambiguity is *"matzah,"* two positions back. Self-attention lets *"ball"* reach directly to *"matzah"* and borrow meaning from it. No chain. No fading memory. A direct connection.

Here is how that works mechanically. Each token is first converted into three vectors:

**Query** — what this token is looking for. *"Ball"* generates a Query that is, roughly, asking: *"Is there a food-type modifier nearby that would tell me what kind of ball I am?"*

**Key** — what this token is advertising. *"Matzah"* generates a Key that says: *"I am a type of food, specifically unleavened bread, relevant to compound nouns."*

**Value** — the actual information to hand over if selected. *"Matzah"*'s Value is its full embedding — the meaning it carries.

The model computes a score between *"ball"*'s Query and every other token's Key. Where Query and Key align, the score is high. *"Ball"* scores high against *"matzah"* and low against *"again"* and *"The."* Those scores are turned into weights with softmax — they add up to 1.0, like a probability distribution over the sentence. Then the model takes a weighted sum of all the Values, with *"matzah"*'s Value contributing most.

The output for *"ball"* is now a blend of the whole sentence, leaning heavily toward *"matzah."* It knows what kind of ball it is.

Mathematically:

<script type="math/tex; mode=display">\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V</script>

The √d scaling keeps the dot products from getting too large and pushing the softmax into a regime where one score dominates everything and the gradients vanish. A technical nuisance, not a deep idea.

<div style="overflow-x:auto;margin:2rem 0;">
<svg viewBox="0 0 680 310" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:auto;font-family:sans-serif;">

  <!-- Input embeddings -->
  <rect x="10" y="120" width="70" height="36" rx="5" fill="#4fb1ba" opacity="0.8"/>
  <text x="45" y="143" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Input x</text>

  <!-- Three projection lines -->
  <line x1="80" y1="128" x2="160" y2="80" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="80" y1="138" x2="160" y2="138" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="80" y1="148" x2="160" y2="200" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,3"/>

  <!-- Q box -->
  <rect x="160" y="62" width="60" height="32" rx="5" fill="#e8a95c"/>
  <text x="190" y="83" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Q</text>
  <text x="190" y="52" text-anchor="middle" fill="#e8a95c" font-size="10">Query</text>

  <!-- K box -->
  <rect x="160" y="122" width="60" height="32" rx="5" fill="#e8a95c"/>
  <text x="190" y="143" text-anchor="middle" fill="white" font-size="12" font-weight="bold">K</text>
  <text x="190" y="112" text-anchor="middle" fill="#e8a95c" font-size="10">Key</text>

  <!-- V box -->
  <rect x="160" y="184" width="60" height="32" rx="5" fill="#e8a95c"/>
  <text x="190" y="205" text-anchor="middle" fill="white" font-size="12" font-weight="bold">V</text>
  <text x="190" y="230" text-anchor="middle" fill="#e8a95c" font-size="10">Value</text>

  <!-- QK dot product -->
  <line x1="220" y1="78" x2="290" y2="120" stroke="#aaa" stroke-width="1.2"/>
  <line x1="220" y1="138" x2="290" y2="138" stroke="#aaa" stroke-width="1.2"/>

  <rect x="290" y="108" width="90" height="40" rx="5" fill="#9b7fd4" opacity="0.85"/>
  <text x="335" y="133" text-anchor="middle" fill="white" font-size="10" font-weight="bold">QKᵀ / √d</text>

  <!-- Softmax -->
  <line x1="380" y1="128" x2="430" y2="128" stroke="#aaa" stroke-width="1.2" marker-end="url(#arr2)"/>
  <rect x="430" y="108" width="80" height="40" rx="5" fill="#9b7fd4" opacity="0.7"/>
  <text x="470" y="133" text-anchor="middle" fill="white" font-size="10" font-weight="bold">softmax</text>
  <text x="470" y="100" text-anchor="middle" fill="#9b7fd4" font-size="10">weights</text>

  <!-- Multiply by V -->
  <line x1="510" y1="128" x2="545" y2="160" stroke="#aaa" stroke-width="1.2"/>
  <line x1="220" y1="200" x2="545" y2="175" stroke="#aaa" stroke-width="1.2" stroke-dasharray="3,3"/>

  <circle cx="555" cy="175" r="14" fill="#4fb1ba" opacity="0.85"/>
  <text x="555" y="180" text-anchor="middle" fill="white" font-size="14" font-weight="bold">×</text>

  <!-- Output -->
  <line x1="569" y1="175" x2="610" y2="175" stroke="#aaa" stroke-width="1.2" marker-end="url(#arr2)"/>
  <rect x="610" y="158" width="60" height="32" rx="5" fill="#4fb1ba"/>
  <text x="640" y="179" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Output</text>

  <defs>
    <marker id="arr2" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#aaa"/>
    </marker>
  </defs>
</svg>
</div>

The key insight: the weights are not fixed. They are computed fresh for every input. A word that is relevant to the current token gets a high weight. A word that is irrelevant gets a low weight. The network *learns* what relevance means.

---

## The Code

In PyTorch, self-attention is a clean computation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        weights = F.softmax(scores, dim=-1)   # (batch, seq_len, seq_len)
        output = torch.bmm(weights, V)         # (batch, seq_len, d_model)

        return output, weights
```

Let us run it on a toy example — five tokens, embedding size 16:

```python
batch_size, seq_len, d_model = 1, 5, 16
x = torch.randn(batch_size, seq_len, d_model)

attn = SelfAttention(d_model)
output, weights = attn(x)

print(f"Input shape:   {x.shape}")       # (1, 5, 16)
print(f"Output shape:  {output.shape}")  # (1, 5, 16)
print(f"Weights shape: {weights.shape}") # (1, 5, 5) — each token attends to all tokens
```

Each token in the output is a blend of *all* tokens in the input. The sequence length is preserved. The information is not bottlenecked into a single vector.

---

## What the Weights Look Like

Below is an illustrative attention map — what a trained model might learn for the sentence *"The bubbe made matzah ball soup again"*:

<iframe src="/assets/charts/attention-heatmap.html" style="width:100%;height:500px;border:none;"></iframe>

Read it row by row. Each row is one query token — the word that is "looking." Each column is a key token — the word being "looked at." Darker means stronger attention.

Notice: *"ball"* attends strongly to *"matzah"* — because "ball" alone means nothing; its meaning depends on what preceded it. *"made"* attends to *"bubbe"* — the verb looks for its subject.

This is not programmed. It is learned.

---

## How Did It Learn That?

*"But wait,"* said Devorah, who had been watching over Mathityahu's shoulder. *"In that chart — 'ball' is already looking at 'matzah.' How does it know to do that? Someone told it?"*

Nobody told it. It learned.

Here is what training looks like. You give the model a task it can be wrong about. The simplest one: **predict the next word**.

Feed in *"The bubbe made matzah ball"* — five tokens. The model produces a probability distribution over the entire vocabulary for what comes next. If it says *"lamp"* and the correct answer is *"soup"*, that is wrong. You measure how wrong — with **cross-entropy loss**:

<script type="math/tex; mode=display">\mathcal{L} = -\log P(\text{soup})</script>

The lower the probability the model assigned to the correct word, the higher the loss. Then you backpropagate — the gradient flows backward through the softmax, through the attention weights, all the way back into the Q, K, and V projection matrices.

At the start of training, those matrices are random. *"ball"* attends to *"The"* just as much as it attends to *"matzah."* The attention map looks like television static.

```python
criterion = nn.CrossEntropyLoss()

# logits: (batch, seq_len, vocab_size) — model's predictions at each position
# targets: (batch, seq_len) — the actual next tokens
logits = model(input_tokens)
loss = criterion(
    logits.view(-1, vocab_size),
    targets.view(-1)
)

loss.backward()
optimizer.step()
```

Over millions of sentences, the gradient keeps nudging the matrices in the same direction: make *"ball"* look at *"matzah,"* because that pattern reliably predicts *"soup."* Make *"made"* look at *"bubbe,"* because verbs need their subjects to predict correctly.

Nobody programmed those relationships. The task demanded them. The loss enforced them.

*"So the attention map is the network's notes,"* said Devorah. *"What it had to learn to get the answers right."*

*"Exactly,"* said Mathityahu. *"And it takes a lot of sentences."*

---

## The Remaining Problem

Rivka could track fourteen conversations, but she knew *who said what when*. The words arrived in order. She knew which story came first.

Self-attention, as written above, has no such knowledge. If you shuffle the tokens — *"soup again bubbe matzah made ball The"* — the attention computation gives exactly the same result. Position has no meaning.

For text, this is catastrophic. "The dog bit the man" must be different from "The man bit the dog." If the model cannot tell position 1 from position 5, it cannot tell subject from object.

We need a way to tell the model: *"this token is first, that one is sixth."* We need to bake position into the representation itself.

And once we solve that — once every token knows both what it is *and* where it sits — we have everything we need to build something much larger.

*"Call everyone in,"* said Mathityahu. *"Multi-head. All of them."*

*[Continue to [Part 3 — The Whole Room Is Listening](/machine-learning/2026-07-03-transformer-encoder-post/)]*

---

*All code in this post runs on CPU with no training required. Full code: [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/).*
