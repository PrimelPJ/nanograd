# nanograd

An automatic differentiation engine built from scratch in TypeScript. No dependencies. Not even a math library.

Every scalar operation builds a computation graph. Call `.backward()` and gradients flow through the entire graph via the chain rule, automatically. Build neurons, layers, and a full MLP on top. Train it on XOR. Watch the loss drop.

```
$ npm run grad-demo

  nanograd — computation graph demo
  ──────────────────────────────────────────────────
  Building: out = tanh(x*w + b)

  forward pass:
    x        = 2
    w        = -3
    b        = 1
    x*w      = -6
    x*w + b  = -5
    tanh(…)  = -0.999909

  gradients after backward():
    d(out)/d(out) = 1   (always 1 — seed)
    d(out)/d(b)   = 0.000182
    d(out)/d(w)   = 0.000364  (x=2 × local_grad)
    d(out)/d(x)   = -0.000546  (w=-3 × local_grad)

  numerical verification:
    d(out)/d(x) numerical ≈ -0.000546
    d(out)/d(w) numerical ≈  0.000364
    d(out)/d(b) numerical ≈  0.000182

  analytical and numerical match — chain rule works.
```

```
$ npm run xor

  epoch    0  loss=0.693142  acc=50%
  epoch  200  loss=0.421087  acc=75%
  epoch  400  loss=0.183204  acc=100%
  epoch  600  loss=0.089341  acc=100%
  epoch  800  loss=0.053201  acc=100%
  epoch 1000  loss=0.036847  acc=100%

  final predictions:
  input      target   predicted   correct
  [0,0]     0        0.0183      Y
  [0,1]     1        0.9821      Y
  [1,0]     1        0.9819      Y
  [1,1]     0        0.0201      Y
```

---

## What it is

When you call `loss.backward()` in PyTorch, it computes gradients for every parameter in your network. nanograd is that mechanism, stripped to its essentials.

The core insight: every arithmetic operation on a `Value` records its inputs and how to compute the local gradient. When you do `x.mul(w).add(b).tanh()`, you're not just computing a number — you're building a directed acyclic graph of every operation. `.backward()` walks that graph in reverse topological order, applying the chain rule at each node to propagate gradients back to the inputs.

That's the entirety of backpropagation. The rest is engineering.

---

## Quick start

```bash
git clone https://github.com/yourusername/nanograd
cd nanograd
npm install
npm run build

npm run grad-demo     # see the computation graph and gradients
npm run xor           # watch a network learn XOR from scratch
npm test              # 21 test cases
```

---

## Architecture

```
Value                        scalar with .data, .grad, ._backward
  │
  ├── arithmetic ops         add, mul, pow, sub, div, neg
  ├── activations            tanh, relu, sigmoid, exp, log
  └── .backward()            reverse topological sort → chain rule

Neuron                       weights[] + bias, forward(inputs) → Value
Layer                        [Neuron × n], forward(inputs) → Value[]
MLP                          [Layer × n], forward(inputs) → Value | Value[]

Loss functions               mseLoss, binaryCrossEntropy, hingeLoss
Training loop                forward → loss → zeroGrad → backward → step
```

### The Value class (`src/value.ts`)

Every scalar in the computation is wrapped in a `Value`. Operations produce new `Value` objects and capture a closure that knows how to push gradients backward:

```typescript
mul(other: Value | number): Value {
  const o = other instanceof Value ? other : new Value(other);
  const out = new Value(this.data * o.data, '', [this, o], '*');

  out._backward = () => {
    this.grad += o.data * out.grad;   // d(out)/d(this) = other
    o.grad += this.data * out.grad;   // d(out)/d(other) = this
  };

  return out;
}
```

The `+=` is intentional — if a node is used multiple times in the graph, its gradient contributions accumulate (multivariate chain rule).

### Backpropagation (`backward()`)

```typescript
backward(): void {
  const topo: Value[] = [];
  const visited = new Set<Value>();

  const buildTopo = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    for (const child of v._prev) buildTopo(child);
    topo.push(v);
  };

  buildTopo(this);
  this.grad = 1.0;  // seed: d(loss)/d(loss) = 1

  for (const v of topo.reverse()) {
    v._backward();  // chain rule at each node
  }
}
```

Post-order DFS gives topological order. Reversed, it gives reverse topological order — every node's output gradient is fully accumulated before its `_backward` fires. This is the only ordering that makes the chain rule work.

### Neural network (`src/nn.ts`)

Built directly on top of `Value`. A `Neuron` is weights and a bias. A `Layer` is a list of neurons. An `MLP` is a list of layers.

```typescript
// A single neuron
forward(inputs: Value[]): Value {
  let act = this.weights
    .map((w, i) => w.mul(inputs[i]))
    .reduce((sum, v) => sum.add(v), this.bias as Value);
  return act.tanh(); // or relu, sigmoid, linear
}
```

There are no matrix operations. Each weight multiplication is a `Value.mul` — every operation is tracked, every gradient is computed.

---

## Why XOR?

XOR is the classic proof-of-need for multi-layer networks. The output isn't linearly separable — no single line (or hyperplane) divides the `0` outputs from the `1` outputs. A single neuron with any activation function cannot learn it.

A two-layer network can. The hidden layer learns to transform the input space into one where the classes *are* linearly separable. Backprop figures out those weights automatically.

```
Input space (not separable):     After hidden layer (separable):
  1 · · · · 0                       0 · · · · · ·
  · · · · · ·                       · · · · · · ·
  · · · · · ·              →        · · · · · · ·
  · · · · · ·                       · 1 · · 1 · ·
  0 · · · · 1                       0 · · · · · ·
```

---

## Gradient checking

Every gradient in the engine is verified against numerical differentiation:

```
numerical ≈ (f(x + h) - f(x - h)) / (2h)
```

Analytical and numerical match to 4 decimal places for all implemented operations. This is how PyTorch's `gradcheck` works internally.

---

## What I learned

Backpropagation is just the chain rule applied in reverse topological order on a computation graph. That's the whole thing. The reason it feels magical in PyTorch is that the graph is built implicitly as you do math — you never see it. Building it explicitly makes the mechanism obvious.

The `+=` on gradients is the part most explanations skip. When a node feeds into two different downstream nodes, it receives gradient contributions from both paths. You accumulate them. This is the multivariate chain rule, not a special case.

---

## What's next

- Batched operations (matrix-valued `Value` for efficiency)
- Adam optimizer (momentum + adaptive learning rate)
- Computation graph visualizer (export to DOT/graphviz)
- Convolutional layer
- Autograd verification suite against PyTorch

---

## Project structure

```
nanograd/
├── src/
│   ├── value.ts       Value class — autograd primitive
│   ├── nn.ts          Neuron, Layer, MLP, loss functions
│   ├── train.ts       training loop utilities
│   └── test.ts        21 test cases
├── examples/
│   ├── xor.ts         XOR training demo
│   └── grad_demo.ts   computation graph walkthrough
├── package.json
└── tsconfig.json
```

---

## References

- Rumelhart, Hinton & Williams (1986). *Learning representations by back-propagating errors.* Nature.
- Karpathy, A. (2022). [micrograd](https://github.com/karpathy/micrograd) — the inspiration for this project.
- Goodfellow, Bengio & Courville. *Deep Learning.* Chapter 6: Deep Feedforward Networks.
- Olah, C. (2015). [Calculus on Computational Graphs: Backpropagation.](https://colah.github.io/posts/2015-08-Backprop/)
