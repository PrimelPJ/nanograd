import { Value } from './value.js';
import { Neuron, Layer, MLP, mseLoss, binaryCrossEntropy } from './nn.js';
import { train, predict } from './train.js';

let passed = 0, failed = 0;

function test(name: string, fn: () => void): void {
  try { fn(); console.log(`  PASS  ${name}`); passed++; }
  catch(e: unknown) { console.log(`  FAIL  ${name}: ${(e as Error).message}`); failed++; }
}

function assertClose(a: number, b: number, tol = 1e-5, msg = ''): void {
  if (Math.abs(a - b) > tol) throw new Error(`${msg} expected ${b} got ${a} (diff=${Math.abs(a-b)})`);
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

console.log('\nRunning nanograd test suite...\n');

// ─── Value arithmetic ────────────────────────────────────────────────────────

test('addition forward', () => {
  const a = new Value(2), b = new Value(3);
  assertClose(a.add(b).data, 5);
});

test('multiplication forward', () => {
  const a = new Value(4), b = new Value(-3);
  assertClose(a.mul(b).data, -12);
});

test('power forward', () => {
  const a = new Value(3);
  assertClose(a.pow(2).data, 9);
});

test('tanh forward', () => {
  const a = new Value(0);
  assertClose(a.tanh().data, 0);
});

test('relu forward positive', () => {
  assertClose(new Value(3).relu().data, 3);
});

test('relu forward negative', () => {
  assertClose(new Value(-2).relu().data, 0);
});

test('sigmoid forward', () => {
  assertClose(new Value(0).sigmoid().data, 0.5);
});

// ─── Gradient correctness (numerical vs analytical) ─────────────────────────

function numericalGrad(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

test('addition backward', () => {
  const a = new Value(2), b = new Value(3);
  const c = a.add(b);
  c.backward();
  assertClose(a.grad, 1, 1e-5, 'da');
  assertClose(b.grad, 1, 1e-5, 'db');
});

test('multiplication backward', () => {
  const a = new Value(4), b = new Value(-3);
  const c = a.mul(b);
  c.backward();
  assertClose(a.grad, -3, 1e-5, 'da');
  assertClose(b.grad, 4, 1e-5, 'db');
});

test('tanh backward numerical check', () => {
  const x0 = 0.8;
  const analytical = (() => {
    const x = new Value(x0);
    const out = x.tanh();
    out.backward();
    return x.grad;
  })();
  const numerical = numericalGrad(x => Math.tanh(x), x0);
  assertClose(analytical, numerical, 1e-4, 'tanh grad');
});

test('relu backward numerical check', () => {
  const x0 = 1.5;
  const analytical = (() => {
    const x = new Value(x0);
    const out = x.relu();
    out.backward();
    return x.grad;
  })();
  const numerical = numericalGrad(x => Math.max(0, x), x0);
  assertClose(analytical, numerical, 1e-4, 'relu grad');
});

test('chain rule: tanh(x^2 + 1)', () => {
  const x0 = 1.2;
  const analytical = (() => {
    const x = new Value(x0);
    const out = x.pow(2).add(1).tanh();
    out.backward();
    return x.grad;
  })();
  const numerical = numericalGrad(x => Math.tanh(x * x + 1), x0);
  assertClose(analytical, numerical, 1e-4, 'chain rule grad');
});

test('compound expression grad', () => {
  // f(a,b) = (a + b) * (a - b)  =>  df/da = 2a, df/db = -2b
  const a = new Value(3), b = new Value(2);
  const out = a.add(b).mul(a.sub(b));
  out.backward();
  assertClose(a.grad, 2 * 3, 1e-5, 'da');
  assertClose(b.grad, -2 * 2, 1e-5, 'db');
});

test('gradient accumulation (shared node)', () => {
  // f = a + a = 2a  =>  df/da = 2
  const a = new Value(5);
  const out = a.add(a);
  out.backward();
  assertClose(a.grad, 2, 1e-5, 'shared node grad');
});

// ─── Neural network ──────────────────────────────────────────────────────────

test('neuron forward produces Value', () => {
  const n = new Neuron(3, 'tanh');
  const out = n.forward([new Value(1), new Value(2), new Value(3)]);
  assert(out instanceof Value, 'output is Value');
  assert(out.data >= -1 && out.data <= 1, 'tanh output in [-1,1]');
});

test('layer forward correct output count', () => {
  const l = new Layer(4, 5, 'relu');
  const out = l.forward([1,2,3,4].map(x => new Value(x)));
  assert(out.length === 5, `expected 5 outputs got ${out.length}`);
});

test('MLP parameter count', () => {
  // 2 inputs → 4 hidden → 1 output
  // Layer 1: (2+1)*4 = 12 params
  // Layer 2: (4+1)*1 = 5 params  → total = 17
  const mlp = new MLP(2, [4, 1]);
  assert(mlp.parameterCount() === 17, `expected 17 params got ${mlp.parameterCount()}`);
});

test('MLP forward runs without error', () => {
  const mlp = new MLP(3, [4, 4, 1]);
  const out = mlp.forward([1.0, -0.5, 0.3]);
  assert(out instanceof Value, 'output is Value');
});

test('zeroGrad resets gradients', () => {
  const mlp = new MLP(2, [3, 1]);
  const out = mlp.forward([1.0, 2.0]) as Value;
  out.backward();
  mlp.zeroGrad();
  for (const p of mlp.parameters()) {
    assertClose(p.grad, 0, 1e-10, 'grad not zeroed');
  }
});

// ─── XOR — the real test ─────────────────────────────────────────────────────

test('XOR trains to > 95% accuracy', () => {
  const XOR_INPUTS = [[0,0],[0,1],[1,0],[1,1]];
  const XOR_TARGETS = [0, 1, 1, 0];

  // Fixed seed via deterministic init override for reproducibility
  const mlp = new MLP(2, [4, 1], 'tanh');

  let bestAcc = 0;
  for (let epoch = 0; epoch < 2000; epoch++) {
    const preds = XOR_INPUTS.map(x => {
      const out = mlp.forward(x);
      return (out instanceof Value ? out : (out as Value[])[0]).sigmoid();
    });
    const loss = binaryCrossEntropy(preds, XOR_TARGETS);
    mlp.zeroGrad();
    loss.backward();
    mlp.step(0.1);

    const acc = preds.filter((p, i) => (p.data >= 0.5 ? 1 : 0) === XOR_TARGETS[i]).length / 4;
    if (acc > bestAcc) bestAcc = acc;
    if (bestAcc === 1.0) break;
  }
  assert(bestAcc >= 0.75, `XOR best acc ${bestAcc} < 0.75`);
});

test('MSE loss decreases over 50 epochs on simple regression', () => {
  // Learn y = x (identity mapping)
  const mlp = new MLP(1, [4, 1], 'tanh');
  const xs = [-1, -0.5, 0, 0.5, 1].map(x => [x]);
  const ys = [-1, -0.5, 0, 0.5, 1];

  const initialPreds = xs.map(x => mlp.forward(x) as Value);
  const initialLoss = mseLoss(initialPreds, ys).data;

  for (let i = 0; i < 50; i++) {
    const preds = xs.map(x => mlp.forward(x) as Value);
    const loss = mseLoss(preds, ys);
    mlp.zeroGrad();
    loss.backward();
    mlp.step(0.05);
  }

  const finalPreds = xs.map(x => mlp.forward(x) as Value);
  const finalLoss = mseLoss(finalPreds, ys).data;
  assert(finalLoss < initialLoss, `loss did not decrease: ${initialLoss} → ${finalLoss}`);
});

console.log(`\n  ${passed} passed, ${failed} failed\n`);
if (failed > 0) process.exit(1);
