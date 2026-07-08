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

But before any of that, the model needs to convert words into numbers. A neural network cannot operate on the word *"matzah"* — it needs a vector. So each word in the vocabulary is mapped to a fixed-length list of numbers, say 512 of them, called an **embedding**. These embeddings are learned during training. Words that appear in similar contexts end up with similar embeddings — *"matzah"* and *"bread"* will be closer together than *"matzah"* and *"accountant."*

So the input to self-attention is not words. It is a sequence of vectors — one per token, each 512 numbers long. For the sentence *"The bubbe made matzah ball soup again,"* that is seven vectors sitting side by side.

Now the attention mechanism has something to work with.

Take the word *"ball."* It needs context. *"Ball"* alone could mean anything — basketball, formal dance, a good time. What resolves the ambiguity is *"matzah,"* two positions back. Self-attention lets *"ball"* reach directly to *"matzah"* and borrow meaning from it. No chain. No fading memory. A direct connection.

Here is how. Each embedding is projected into three new vectors through learned weight matrices:

**Query** — what this token is looking for. *"Ball"* generates a Query that is, roughly, asking: *"Is there a food-type modifier nearby that would tell me what kind of ball I am?"*

**Key** — what this token is advertising. *"Matzah"* generates a Key that says: *"I am a type of food, specifically unleavened bread, relevant to compound nouns."*

**Value** — the actual information to hand over if selected. *"Matzah"*'s Value is its full embedding — the meaning it carries.

The model computes a score between *"ball"*'s Query and every other token's Key. Where Query and Key align, the score is high. *"Ball"* scores high against *"matzah"* and low against *"again"* and *"The."* Those scores are turned into weights with softmax — they add up to 1.0, like a probability distribution over the sentence. Then the model takes a weighted sum of all the Values, with *"matzah"*'s Value contributing most.

The output for *"ball"* is now a blend of the whole sentence, leaning heavily toward *"matzah."* It knows what kind of ball it is.

Mathematically, for a single head:

<script type="math/tex; mode=display">\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V</script>

The √d scaling keeps the dot products from getting too large and pushing the softmax into a regime where one score dominates everything and the gradients vanish. A technical nuisance, not a deep idea.

<div style="overflow-x:auto;margin:2rem 0;">
<svg viewBox="0 0 500 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:auto;font-family:sans-serif;">
  <defs>
    <marker id="sha" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto">
      <path d="M0,7 L3.5,0 L7,7 z" fill="#aaa"/>
    </marker>
  </defs>

  <!-- OUTPUT Z -->
  <rect x="200" y="8" width="100" height="28" rx="5" fill="#4fb1ba"/>
  <text x="250" y="27" text-anchor="middle" fill="white" font-size="13" font-weight="bold">Z</text>

  <!-- Attention → Z -->
  <line x1="250" y1="68" x2="250" y2="36" stroke="#bbb" stroke-width="1.5" marker-end="url(#sha)"/>

  <!-- ATTENTION BLOCK -->
  <rect x="60" y="68" width="380" height="36" rx="5" fill="#9b7fd4" opacity="0.75"/>
  <text x="250" y="83" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Scaled Dot-Product Attention</text>
  <text x="250" y="97" text-anchor="middle" fill="white" font-size="10">softmax( QKᵀ / √d ) · V</text>

  <!-- Q K V labels -->
  <line x1="130" y1="104" x2="130" y2="120" stroke="#bbb" stroke-width="1.1"/>
  <line x1="250" y1="104" x2="250" y2="120" stroke="#bbb" stroke-width="1.1"/>
  <line x1="370" y1="104" x2="370" y2="120" stroke="#bbb" stroke-width="1.1"/>
  <text x="130" y="130" text-anchor="middle" fill="#e8a95c" font-size="12" font-weight="bold">Q</text>
  <text x="250" y="130" text-anchor="middle" fill="#e8a95c" font-size="12" font-weight="bold">K</text>
  <text x="370" y="130" text-anchor="middle" fill="#e8a95c" font-size="12" font-weight="bold">V</text>

  <!-- Linear boxes -->
  <line x1="130" y1="132" x2="130" y2="142" stroke="#bbb" stroke-width="1.1" marker-end="url(#sha)"/>
  <rect x="80" y="142" width="100" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="130" y="154" text-anchor="middle" fill="white" font-size="10" font-weight="bold">W_Q</text>
  <text x="130" y="164" text-anchor="middle" fill="white" font-size="9">Linear</text>

  <line x1="250" y1="132" x2="250" y2="142" stroke="#bbb" stroke-width="1.1" marker-end="url(#sha)"/>
  <rect x="200" y="142" width="100" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="250" y="154" text-anchor="middle" fill="white" font-size="10" font-weight="bold">W_K</text>
  <text x="250" y="164" text-anchor="middle" fill="white" font-size="9">Linear</text>

  <line x1="370" y1="132" x2="370" y2="142" stroke="#bbb" stroke-width="1.1" marker-end="url(#sha)"/>
  <rect x="320" y="142" width="100" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="370" y="154" text-anchor="middle" fill="white" font-size="10" font-weight="bold">W_V</text>
  <text x="370" y="164" text-anchor="middle" fill="white" font-size="9">Linear</text>

  <!-- Bus to X -->
  <line x1="130" y1="169" x2="130" y2="220" stroke="#bbb" stroke-width="1.1"/>
  <line x1="250" y1="169" x2="250" y2="220" stroke="#bbb" stroke-width="1.1"/>
  <line x1="370" y1="169" x2="370" y2="220" stroke="#bbb" stroke-width="1.1"/>
  <line x1="130" y1="220" x2="370" y2="220" stroke="#bbb" stroke-width="1.5"/>

  <!-- X input -->
  <line x1="250" y1="248" x2="250" y2="222" stroke="#bbb" stroke-width="1.5" marker-end="url(#sha)"/>
  <rect x="190" y="248" width="120" height="28" rx="5" fill="#4fb1ba"/>
  <text x="250" y="267" text-anchor="middle" fill="white" font-size="13" font-weight="bold">X</text>
  <text x="250" y="290" text-anchor="middle" fill="#888" font-size="10">input embeddings</text>
</svg>
</div>

One attention head produces one perspective on the sentence — one particular notion of what is relevant to what. That is already powerful. But language carries multiple kinds of relationships simultaneously. *"The bubbe made matzah ball soup again"* — one head might notice that *"made"* attends to *"bubbe"* (verb to subject). Another might notice that *"ball"* attends to *"matzah"* (noun to modifier). A third might notice that *"again"* attends to *"made"* (adverb to the verb it modifies). No single head can learn all of this at once without the relationships interfering with each other.

*"So run it multiple times,"* said Devorah.

Exactly. **Multi-head attention** runs h independent attention operations in parallel, each with its own W_Q, W_K, W_V matrices. Each head attends to the sentence through a different learned lens. Then the h outputs are concatenated and passed through one final linear layer W_O to produce the result.

<div style="overflow-x:auto;margin:2rem 0;">
<svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;display:block;margin:auto;font-family:sans-serif;">
  <defs>
    <marker id="mha" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto">
      <path d="M0,7 L3.5,0 L7,7 z" fill="#aaa"/>
    </marker>
  </defs>

  <!-- OUTPUT -->
  <rect x="300" y="8" width="100" height="28" rx="5" fill="#4fb1ba"/>
  <text x="350" y="27" text-anchor="middle" fill="white" font-size="13" font-weight="bold">Z</text>
  <text x="415" y="26" fill="#888" font-size="10" font-style="italic">output</text>

  <!-- W_O → Z -->
  <line x1="350" y1="63" x2="350" y2="36" stroke="#bbb" stroke-width="1.5" marker-end="url(#mha)"/>

  <!-- W_O -->
  <rect x="185" y="63" width="330" height="27" rx="5" fill="#e8a95c"/>
  <text x="350" y="81" text-anchor="middle" fill="white" font-size="11" font-weight="bold">W_O · Linear</text>

  <!-- Concat → W_O -->
  <line x1="350" y1="112" x2="350" y2="90" stroke="#bbb" stroke-width="1.5" marker-end="url(#mha)"/>

  <!-- CONCATENATION -->
  <rect x="22" y="112" width="656" height="27" rx="5" fill="#9b7fd4" opacity="0.82"/>
  <text x="350" y="130" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Concatenate  [ Z₁ ,  Z₂ ,  · · · ,  Zₕ ]</text>

  <!-- Z1 label and line -->
  <line x1="145" y1="153" x2="145" y2="139" stroke="#bbb" stroke-width="1.2" marker-end="url(#mha)"/>
  <text x="145" y="163" text-anchor="middle" fill="#9b7fd4" font-size="12" font-weight="bold">Z₁</text>

  <!-- Zh label and line -->
  <line x1="555" y1="153" x2="555" y2="139" stroke="#bbb" stroke-width="1.2" marker-end="url(#mha)"/>
  <text x="555" y="163" text-anchor="middle" fill="#9b7fd4" font-size="12" font-weight="bold">Zₕ</text>

  <!-- dots between Z labels -->
  <text x="350" y="162" text-anchor="middle" fill="#ccc" font-size="18">· · ·</text>

  <!-- ATTENTION HEAD 1 → Z1 -->
  <line x1="145" y1="175" x2="145" y2="166" stroke="#bbb" stroke-width="1.2" marker-end="url(#mha)"/>

  <!-- ATTENTION BLOCK HEAD 1 -->
  <rect x="22" y="175" width="246" height="36" rx="5" fill="#9b7fd4" opacity="0.6"/>
  <text x="145" y="190" text-anchor="middle" fill="white" font-size="10" font-weight="bold">Scaled Dot-Product</text>
  <text x="145" y="204" text-anchor="middle" fill="white" font-size="10">Attention — head 1</text>

  <!-- ATTENTION HEAD h → Zh -->
  <line x1="555" y1="175" x2="555" y2="166" stroke="#bbb" stroke-width="1.2" marker-end="url(#mha)"/>

  <!-- ATTENTION BLOCK HEAD h -->
  <rect x="432" y="175" width="246" height="36" rx="5" fill="#9b7fd4" opacity="0.6"/>
  <text x="555" y="190" text-anchor="middle" fill="white" font-size="10" font-weight="bold">Scaled Dot-Product</text>
  <text x="555" y="204" text-anchor="middle" fill="white" font-size="10">Attention — head h</text>

  <!-- dots between attention blocks -->
  <text x="350" y="198" text-anchor="middle" fill="#ccc" font-size="18">· · ·</text>

  <!-- Q K V labels head 1 -->
  <line x1="63" y1="211" x2="63" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <line x1="145" y1="211" x2="145" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <line x1="227" y1="211" x2="227" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <text x="63"  y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">Q₁</text>
  <text x="145" y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">K₁</text>
  <text x="227" y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">V₁</text>

  <!-- Q K V labels head h -->
  <line x1="473" y1="211" x2="473" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <line x1="555" y1="211" x2="555" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <line x1="637" y1="211" x2="637" y2="224" stroke="#bbb" stroke-width="1.1"/>
  <text x="473" y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">Qₕ</text>
  <text x="555" y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">Kₕ</text>
  <text x="637" y="234" text-anchor="middle" fill="#e8a95c" font-size="11" font-weight="bold">Vₕ</text>

  <!-- LINEAR BOXES HEAD 1 -->
  <line x1="63"  y1="236" x2="63"  y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="22"  y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="63"  y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_Q,1</text>
  <text x="63"  y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <line x1="145" y1="236" x2="145" y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="104" y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="145" y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_K,1</text>
  <text x="145" y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <line x1="227" y1="236" x2="227" y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="186" y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="227" y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_V,1</text>
  <text x="227" y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <!-- LINEAR BOXES HEAD h -->
  <line x1="473" y1="236" x2="473" y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="432" y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="473" y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_Q,h</text>
  <text x="473" y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <line x1="555" y1="236" x2="555" y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="514" y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="555" y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_K,h</text>
  <text x="555" y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <line x1="637" y1="236" x2="637" y2="244" stroke="#bbb" stroke-width="1.1" marker-end="url(#mha)"/>
  <rect x="596" y="244" width="82" height="27" rx="4" fill="#e8a95c" opacity="0.85"/>
  <text x="637" y="256" text-anchor="middle" fill="white" font-size="9" font-weight="bold">W_V,h</text>
  <text x="637" y="266" text-anchor="middle" fill="white" font-size="8">Linear</text>

  <!-- dots between linear groups -->
  <text x="350" y="261" text-anchor="middle" fill="#ccc" font-size="18">· · ·</text>

  <!-- Vertical drops from linear boxes to bus -->
  <line x1="63"  y1="271" x2="63"  y2="312" stroke="#bbb" stroke-width="1.1"/>
  <line x1="145" y1="271" x2="145" y2="312" stroke="#bbb" stroke-width="1.1"/>
  <line x1="227" y1="271" x2="227" y2="312" stroke="#bbb" stroke-width="1.1"/>
  <line x1="473" y1="271" x2="473" y2="312" stroke="#bbb" stroke-width="1.1"/>
  <line x1="555" y1="271" x2="555" y2="312" stroke="#bbb" stroke-width="1.1"/>
  <line x1="637" y1="271" x2="637" y2="312" stroke="#bbb" stroke-width="1.1"/>

  <!-- Horizontal bus -->
  <line x1="63" y1="312" x2="637" y2="312" stroke="#bbb" stroke-width="1.5"/>

  <!-- X input box -->
  <line x1="350" y1="340" x2="350" y2="314" stroke="#bbb" stroke-width="1.5" marker-end="url(#mha)"/>
  <rect x="295" y="340" width="110" height="28" rx="5" fill="#4fb1ba"/>
  <text x="350" y="359" text-anchor="middle" fill="white" font-size="13" font-weight="bold">X</text>
  <text x="350" y="382" text-anchor="middle" fill="#888" font-size="10">input embeddings</text>
</svg>
</div>

The key insight: the weights are not fixed. They are computed fresh for every input. Each head learns to look for *different* kinds of relationships — one head might learn syntax (verbs attending to their subjects), another semantics (nouns attending to modifiers). The network *learns* what relevance means, independently, in parallel, eight times over.

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
