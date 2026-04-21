// Training utilities and built-in training loop

import { Value } from './value.js';
import { MLP, mseLoss, binaryCrossEntropy } from './nn.js';

export interface TrainConfig {
  epochs: number;
  learningRate: number;
  printEvery?: number;
  lrDecay?: number;
}

export interface TrainResult {
  losses: number[];
  finalLoss: number;
  epochs: number;
}

export function train(
  model: MLP,
  inputs: number[][],
  targets: number[],
  config: TrainConfig
): TrainResult {
  const { epochs, learningRate, printEvery = 100, lrDecay = 1.0 } = config;
  const losses: number[] = [];
  let lr = learningRate;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Forward pass
    const preds = inputs.map(x => {
      const out = model.forward(x);
      return out instanceof Value ? out : (out as Value[])[0];
    });

    // Apply sigmoid for binary classification
    const sigPreds = preds.map(p => p.sigmoid());
    const loss = binaryCrossEntropy(sigPreds, targets);

    // Backward pass
    model.zeroGrad();
    loss.backward();

    // SGD update
    lr = learningRate * Math.pow(lrDecay, epoch);
    model.step(lr);

    losses.push(loss.data);

    if (printEvery > 0 && (epoch % printEvery === 0 || epoch === epochs - 1)) {
      const acc = computeAccuracy(sigPreds, targets);
      console.log(`  epoch ${String(epoch).padStart(4)}  loss=${loss.data.toFixed(6)}  acc=${(acc * 100).toFixed(1)}%  lr=${lr.toFixed(6)}`);
    }
  }

  return { losses, finalLoss: losses[losses.length - 1], epochs };
}

export function computeAccuracy(predictions: Value[], targets: number[]): number {
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i].data >= 0.5 ? 1 : 0;
    const target = targets[i] >= 0.5 ? 1 : 0;
    if (pred === target) correct++;
  }
  return correct / predictions.length;
}

export function predict(model: MLP, input: number[]): number {
  const out = model.forward(input);
  const val = out instanceof Value ? out : (out as Value[])[0];
  return val.sigmoid().data;
}
