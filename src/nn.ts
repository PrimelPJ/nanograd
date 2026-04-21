// Neural network primitives built on top of Value
// Neuron → Layer → MLP — same structure as PyTorch nn.Module

import { Value } from './value.js';

function randn(): number {
  // Box-Muller transform for Gaussian init
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export class Neuron {
  weights: Value[];
  bias: Value;
  activation: Activation;

  constructor(numInputs: number, activation: Activation = 'tanh') {
    // He initialization for relu, Xavier-ish for tanh/sigmoid
    const scale = activation === 'relu' ? Math.sqrt(2 / numInputs) : Math.sqrt(1 / numInputs);
    this.weights = Array.from({ length: numInputs }, (_, i) => {
      const v = new Value(randn() * scale);
      v.label = `w${i}`;
      return v;
    });
    this.bias = new Value(0);
    this.bias.label = 'b';
    this.activation = activation;
  }

  forward(inputs: Value[]): Value {
    if (inputs.length !== this.weights.length) {
      throw new Error(`Expected ${this.weights.length} inputs, got ${inputs.length}`);
    }
    // Weighted sum: w·x + b
    let act = this.weights
      .map((w, i) => w.mul(inputs[i]))
      .reduce((sum, v) => sum.add(v), this.bias as Value);

    // Apply activation
    switch (this.activation) {
      case 'tanh':    return act.tanh();
      case 'relu':    return act.relu();
      case 'sigmoid': return act.sigmoid();
      case 'linear':  return act;
    }
  }

  parameters(): Value[] {
    return [...this.weights, this.bias];
  }

  parameterCount(): number {
    return this.weights.length + 1;
  }
}

export class Layer {
  neurons: Neuron[];

  constructor(numInputs: number, numOutputs: number, activation: Activation = 'tanh') {
    this.neurons = Array.from({ length: numOutputs }, () => new Neuron(numInputs, activation));
  }

  forward(inputs: Value[]): Value[] {
    return this.neurons.map(n => n.forward(inputs));
  }

  parameters(): Value[] {
    return this.neurons.flatMap(n => n.parameters());
  }

  parameterCount(): number {
    return this.neurons.reduce((s, n) => s + n.parameterCount(), 0);
  }
}

export class MLP {
  layers: Layer[];
  private arch: number[];

  constructor(numInputs: number, layerSizes: number[], activation: Activation = 'tanh') {
    this.arch = [numInputs, ...layerSizes];
    this.layers = layerSizes.map((size, i) => {
      const isLast = i === layerSizes.length - 1;
      // Last layer uses linear for regression, sigmoid for binary classification
      return new Layer(this.arch[i], size, isLast ? 'linear' : activation);
    });
  }

  forward(inputs: (Value | number)[]): Value | Value[] {
    let current: Value[] = inputs.map(x => x instanceof Value ? x : new Value(x));
    for (const layer of this.layers) {
      current = layer.forward(current);
    }
    return current.length === 1 ? current[0] : current;
  }

  parameters(): Value[] {
    return this.layers.flatMap(l => l.parameters());
  }

  parameterCount(): number {
    return this.layers.reduce((s, l) => s + l.parameterCount(), 0);
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.grad = 0;
  }

  // SGD step
  step(lr: number): void {
    for (const p of this.parameters()) {
      p.data -= lr * p.grad;
    }
  }

  toString(): string {
    const arch = this.arch.join(' → ');
    return `MLP(${arch}) — ${this.parameterCount()} parameters`;
  }
}

// ─── Loss functions ──────────────────────────────────────────────────────────

export function mseLoss(predictions: Value[], targets: number[]): Value {
  if (predictions.length !== targets.length) throw new Error('Length mismatch');
  const losses = predictions.map((p, i) => p.sub(targets[i]).pow(2));
  return losses.reduce((sum, l) => sum.add(l)).div(predictions.length);
}

export function binaryCrossEntropy(predictions: Value[], targets: number[]): Value {
  if (predictions.length !== targets.length) throw new Error('Length mismatch');
  const eps = 1e-7;
  const losses = predictions.map((p, i) => {
    const t = targets[i];
    // Clamp to avoid log(0)
    const pClamped = p.add(eps).div(1 + 2 * eps);
    if (t === 1) return pClamped.log().neg();
    return pClamped.neg().add(1).log().neg();
  });
  return losses.reduce((sum, l) => sum.add(l)).div(predictions.length);
}

export function hingeLoss(predictions: Value[], targets: number[]): Value {
  // targets should be -1 or +1
  const losses = predictions.map((p, i) => {
    const margin = p.mul(targets[i]);
    const zero = new Value(0);
    // max(0, 1 - y*pred) — approximate with relu
    return new Value(1).sub(margin).relu();
  });
  return losses.reduce((sum, l) => sum.add(l)).div(predictions.length);
}
