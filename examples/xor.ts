// Classic XOR demo — the problem a single neuron can't solve
// A 2-layer network learns it from scratch using backprop

import { MLP, binaryCrossEntropy } from '../src/nn.js';
import { Value } from '../src/value.js';

const XOR_INPUTS = [[0,0],[0,1],[1,0],[1,1]];
const XOR_TARGETS = [0, 1, 1, 0];

console.log('\n  nanograd — XOR demo');
console.log('  ' + '─'.repeat(50));
console.log('  A single neuron cannot learn XOR.');
console.log('  A 2-layer MLP with backprop can.\n');

const model = new MLP(2, [4, 1], 'tanh');
console.log(`  model: ${model}`);
console.log(`  training for 1000 epochs, lr=0.1\n`);

for (let epoch = 0; epoch <= 1000; epoch++) {
  const preds = XOR_INPUTS.map(x => {
    const out = model.forward(x);
    return (out instanceof Value ? out : (out as Value[])[0]).sigmoid();
  });

  const loss = binaryCrossEntropy(preds, XOR_TARGETS);
  model.zeroGrad();
  loss.backward();
  model.step(0.1);

  if (epoch % 200 === 0) {
    const acc = preds.filter((p,i) => (p.data >= 0.5 ? 1 : 0) === XOR_TARGETS[i]).length / 4;
    console.log(`  epoch ${String(epoch).padStart(4)}  loss=${loss.data.toFixed(6)}  acc=${(acc*100).toFixed(0)}%`);
  }
}

console.log('\n  final predictions:');
console.log('  ' + '─'.repeat(50));
console.log('  input      target   predicted   correct');
XOR_INPUTS.forEach((x, i) => {
  const out = model.forward(x);
  const val = (out instanceof Value ? out : (out as Value[])[0]).sigmoid();
  const pred = val.data >= 0.5 ? 1 : 0;
  const correct = pred === XOR_TARGETS[i] ? 'Y' : 'N';
  console.log(`  [${x}]     ${XOR_TARGETS[i]}        ${val.data.toFixed(4)}      ${correct}`);
});
console.log();
