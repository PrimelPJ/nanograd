// Value — the core autograd primitive
// Every scalar in the computation graph is a Value.
// Operations build a DAG; backward() walks it in reverse topological order.

type Op = '+' | '*' | '**' | 'tanh' | 'relu' | 'sigmoid' | 'exp' | 'log' | '';

export class Value {
  data: number;
  grad: number = 0;
  label: string;

  private _backward: () => void = () => {};
  private _prev: Set<Value>;
  private _op: Op;

  constructor(data: number, label = '', prev: Value[] = [], op: Op = '') {
    this.data = data;
    this.label = label;
    this._prev = new Set(prev);
    this._op = op;
  }

  // ─── Arithmetic ────────────────────────────────────────────────────────────

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, '', [this, o], '+');
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, '', [this, o], '*');
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(exp: number): Value {
    const out = new Value(Math.pow(this.data, exp), '', [this], '**');
    out._backward = () => {
      this.grad += exp * Math.pow(this.data, exp - 1) * out.grad;
    };
    return out;
  }

  neg(): Value { return this.mul(-1); }
  sub(other: Value | number): Value { return this.add(other instanceof Value ? other.neg() : -other); }
  div(other: Value | number): Value { return this.mul(other instanceof Value ? other.pow(-1) : 1 / (other as number)); }

  // ─── Activations ──────────────────────────────────────────────────────────

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, '', [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(Math.max(0, this.data), '', [this], 'relu');
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, '', [this], 'sigmoid');
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, '', [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    if (this.data <= 0) throw new Error('log of non-positive value');
    const out = new Value(Math.log(this.data), '', [this], 'log');
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  // ─── Backward pass ────────────────────────────────────────────────────────

  backward(): void {
    // Build topological order via DFS
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._prev) buildTopo(child);
      topo.push(v);
    };

    buildTopo(this);
    this.grad = 1.0;

    // Walk in reverse — chain rule propagates gradients backward
    for (const v of topo.reverse()) {
      v._backward();
    }
  }

  // Zero all gradients in the graph (call before each forward pass)
  zeroGrad(): void {
    const visited = new Set<Value>();
    const zero = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      v.grad = 0;
      for (const child of v._prev) zero(child);
    };
    zero(this);
  }

  // ─── Graph inspection ─────────────────────────────────────────────────────

  // Returns all nodes and edges in the computation graph
  trace(): { nodes: Value[]; edges: [Value, Value][] } {
    const nodes: Value[] = [];
    const edges: [Value, Value][] = [];
    const visited = new Set<Value>();

    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      nodes.push(v);
      for (const child of v._prev) {
        edges.push([child, v]);
        build(child);
      }
    };
    build(this);
    return { nodes, edges };
  }

  // Print the computation graph as ASCII
  printGraph(indent = 0): void {
    const pad = '  '.repeat(indent);
    const op = this._op ? `[${this._op}]` : '';
    const label = this.label ? `(${this.label})` : '';
    console.log(`${pad}${label}${op} data=${this.data.toFixed(4)} grad=${this.grad.toFixed(4)}`);
    for (const child of this._prev) child.printGraph(indent + 1);
  }

  toString(): string {
    return `Value(data=${this.data.toFixed(6)}, grad=${this.grad.toFixed(6)})`;
  }
}
