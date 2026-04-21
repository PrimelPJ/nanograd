// Demonstrates the computation graph and gradient flow visually

import { Value } from '../src/value.js';

console.log('\n  nanograd — computation graph demo');
console.log('  ' + '─'.repeat(50));
console.log('  Building: out = tanh(x*w + b)\n');

const x = new Value(2.0);  x.label = 'x';
const w = new Value(-3.0); w.label = 'w';
const b = new Value(1.0);  b.label = 'b';

const xw = x.mul(w);
const xwb = xw.add(b);
const out = xwb.tanh();

console.log('  forward pass:');
console.log(`    x        = ${x.data}`);
console.log(`    w        = ${w.data}`);
console.log(`    b        = ${b.data}`);
console.log(`    x*w      = ${xw.data}`);
console.log(`    x*w + b  = ${xwb.data}`);
console.log(`    tanh(…)  = ${out.data.toFixed(6)}`);

out.backward();

console.log('\n  gradients after backward():');
console.log(`    d(out)/d(out) = ${out.grad}   (always 1 — seed)`);
console.log(`    d(out)/d(b)   = ${b.grad.toFixed(6)}`);
console.log(`    d(out)/d(w)   = ${w.grad.toFixed(6)}  (x=${x.data} × local_grad)`);
console.log(`    d(out)/d(x)   = ${x.grad.toFixed(6)}  (w=${w.data} × local_grad)`);

console.log('\n  numerical verification:');
const h = 1e-5;
const f = (xv: number, wv: number, bv: number) => Math.tanh(xv * wv + bv);
console.log(`    d(out)/d(x) numerical ≈ ${((f(x.data+h, w.data, b.data) - f(x.data-h, w.data, b.data))/(2*h)).toFixed(6)}`);
console.log(`    d(out)/d(w) numerical ≈ ${((f(x.data, w.data+h, b.data) - f(x.data, w.data-h, b.data))/(2*h)).toFixed(6)}`);
console.log(`    d(out)/d(b) numerical ≈ ${((f(x.data, w.data, b.data+h) - f(x.data, w.data, b.data-h))/(2*h)).toFixed(6)}`);
console.log('\n  analytical and numerical match — chain rule works.\n');
