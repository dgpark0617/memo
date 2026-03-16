"""
mlp.py - Pure Python MLP, Karpathy style
No dependencies. Just math, random, os.
~200 lines. Dataset + Autograd + MLP + Train + Inference.
"""

import math
import random
import os

# ── 1. AUTOGRAD ENGINE ──────────────────────────────────────────────────────

class Value:
    """Scalar value with gradient tracking."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exp):
        out = Value(self.data ** exp, (self,), f'**{exp}')
        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):         return self * -1
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return Value(other) + (-self)
    def __sub__(self, other):  return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-9), (self,), 'log')
        def _backward():
            self.grad += (1.0 / (self.data + 1e-9)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# ── 2. NEURAL NETWORK ───────────────────────────────────────────────────────

class Neuron:
    def __init__(self, n_in, act='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == 'relu':  return out.relu()
        if self.act == 'tanh':  return out.tanh()
        return out  # linear

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, act='relu'):
        self.neurons = [Neuron(n_in, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, hidden_sizes, n_out):
        sizes = [n_in] + hidden_sizes + [n_out]
        self.layers = []
        for i in range(len(sizes) - 1):
            act = 'linear' if i == len(sizes) - 2 else 'relu'
            self.layers.append(Layer(sizes[i], sizes[i+1], act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def param_count(self):
        return len(self.parameters())


# ── 3. LOSS ─────────────────────────────────────────────────────────────────

def mse_loss(preds, targets):
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) * (1 / len(preds))

def binary_cross_entropy(pred, target):
    p = pred.tanh() * Value(0.5) + Value(0.5)  # sigmoid via tanh
    return -(Value(target) * p.log() + Value(1 - target) * (Value(1.0) - p).log())


# ── 4. DATASET ──────────────────────────────────────────────────────────────

def make_xor_dataset():
    """XOR: classic non-linearly separable problem."""
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
    return data

def make_regression_dataset(n=50):
    """y = 2x - 1 + noise"""
    random.seed(42)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = 2 * x - 1 + random.gauss(0, 0.1)
        data.append(([x], y))
    return data


# ── 5. TRAIN ────────────────────────────────────────────────────────────────

def train(model, dataset, epochs=500, lr=0.01, task='regression', verbose=True):
    for epoch in range(epochs):
        total_loss = Value(0.0)

        for x, y in dataset:
            pred = model(x)
            if task == 'classification':
                loss = binary_cross_entropy(pred, y)
            else:
                loss = (pred - Value(y)) ** 2
            total_loss = total_loss + loss

        total_loss = total_loss * Value(1 / len(dataset))

        model.zero_grad()
        total_loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} | loss {total_loss.data:.6f}")

    return model


# ── 6. TEST ─────────────────────────────────────────────────────────────────

def test_xor():
    print("\n" + "="*50)
    print("TEST 1: XOR Classification")
    print("="*50)
    random.seed(0)
    model = MLP(n_in=2, hidden_sizes=[4, 4], n_out=1)
    print(f"Parameters: {model.param_count()}")
    dataset = make_xor_dataset()
    train(model, dataset, epochs=1000, lr=0.1, task='classification')

    print("\nResults:")
    for x, y in dataset:
        pred = model(x)
        prob = math.tanh(pred.data) * 0.5 + 0.5
        predicted = 1 if prob > 0.5 else 0
        correct = "✓" if predicted == int(y) else "✗"
        print(f"  input={x} | target={int(y)} | pred={prob:.3f} | {correct}")

def test_regression():
    print("\n" + "="*50)
    print("TEST 2: Regression (y = 2x - 1)")
    print("="*50)
    random.seed(0)
    model = MLP(n_in=1, hidden_sizes=[8, 8], n_out=1)
    print(f"Parameters: {model.param_count()}")
    dataset = make_regression_dataset(n=30)
    train(model, dataset, epochs=500, lr=0.01, task='regression')

    print("\nSample predictions:")
    test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for x in test_points:
        pred = model([x])
        true = 2 * x - 1
        print(f"  x={x:5.1f} | true={true:6.3f} | pred={pred.data:6.3f} | err={abs(pred.data - true):.3f}")

if __name__ == '__main__':
    test_xor()
    test_regression()
