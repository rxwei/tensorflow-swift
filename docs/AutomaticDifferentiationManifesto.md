# The Automatic Differentiation Manifesto

Author: [Richard Wei](https://github.com/rxwei)
Date: October 2018

## Introduction

Automatic Differentiation (AD), also known as algorithmic differentiation, is a
family of techniques used to obtain the derivative of a function. Functions can
be represented as a composition of elementary operators whose derivatives are
well-known. While partial derivatives can be computed through different
techniques, the most common is a recursive application of the chain rule in the
reverse direction, called reverse-mode AD. Reverse-mode AD computes
vector-Jacobian products, i.e. partial derivatives with respect to each input
parameter, and it has become a prerequisite for implementing gradient-based
learning methods. AD has a rich background, here are some great introductions:
[Introduction to Automatic
Differentiation](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/)
and [Automatic Differentiation in Machine Learning: a
Survey](https://arxiv.org/abs/1502.05767).

Most AD implementations work on a graph representation of a functional tensor
program, and many have limited expressivity and extensibility. Frameworks based
on a define-by-run programming model (to support dynamic computation graphs)
often lack the ability to perform full-program static analysis and
optimizations, and make it hard to diagnose errors and target hardware
accelerators ahead of time.

We aim to provide best-in-class AD, including the best optimizations, best error
messages in failure cases, and the most flexibility and expressivity. To achieve
this, we built support for AD right into the Swift compiler. Additionally, since
AD is important to the broader scientific and numerical computing communities,
we decided to build AD as a generic feature that is completely orthogonal to the
TensorFlow support - the TensorFlow Swift library computes gradients using the
AD features of the Swift language itself.

## Why Swift needs Automatic Differentiation?


## Vision

By integrating automatic differentiation directly into the programming language.

## System Design

### Differentiable Types

#### Standard Library Extensions

### Primitive Derivatives

```
declaration-attribute-list ::= differentiable-attribute declaration-attributes
differentiable-attribute ::= '@differentiable' '(' differentiability ')'
differentiability ::= differentiation-mode | 'bidirectional' | 'linear' | 'constant'
differentiation-mode ::= 'forward' | 'reverse'
```

### Raw Differential Operators

```
expression ::= gradient-expression
derivative-expression ::= '#derivative' '(' expression ')'
gradient-expression ::= '#gradient' '(' expression ')'
value-and-derivative-expression ::= '#valueAndDerivative' '(' expression ')'
value-and-gradient-expression ::= '#valueAndGradient' '(' expression ')'
chainable-derivative-expression ::= '#chainableDerivative' '(' expression ')'
chainable-gradient-expression ::= '#chainableGradient' '(' expression ')'
```

Example:
```swift
func foo(x: Float) -> Float {
    return sin(x + y)
}
let dfoo = #derivative(foo)
let 𝝯foo = #gradient(foo)
```

### Generalized Differentiability


### Differentiation API

1. Directional derivatives and gradients

```swift
/// Computes the derivative of the trailing closure at `x`.
func derivative<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(forward) (T) throws -> R
) rethrows -> R.TangentVector {
    return #derivative(body)(x)
}

/// Computes the directional derivative at `x` along the direction `u`.
func directionalDerivative<T : Differentiable, R : Differentiable>(
    at x: T, along direction: T.TangentVector, in body: @autodiff(forward) (T) throws -> R
) rethrows -> R.TangentVector {
    return #chainableDerivative(body)(x, direction)
}

/// Computes the gradient of the trailing closure at `x`.
func gradient<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> T.CotangentVector {
    return #gradient(body)(x)
}

/// Computes the gradient of some function through the trailing closure at `x`,
/// backpropagating the given value.
func gradient<T : Differentiable, R : Differentiable>(
    at x: T, backpropagating seed: R.CotangentVector, 
    in body: @autodiff(reverse) (T) throws -> R
) rethrows -> T.CotangentVector {
    return #chainableGradient(body, seed)(x)
}

/// Computes the gradient of the trailing closure at `x` and `y`.
func gradient<T : Differentiable, U : Differentiable, R : Differentiable>(
    at x: T, _ y: U, 
    in body: @autodiff(reverse) (T, U) throws -> R
) rethrows -> (T.CotangentVector, U.CotangentVector) {
    return #gradient(body)(x, y)
}

/// Computes the gradient of the trailing closure at `x`, `y`, and `z`.
func gradient<T : Differentiable, U : Differentiable, V : Differentiable, R : Differentiable>(
    wrt x: T, _ y: U, _ z: V,
    in body: @autodiff(reverse) (T, U, V) throws -> R
) rethrows -> (T.CotangentVector, U.CotangentVector, V.CotangentVector) {
    return #gradient(body)(x, y, z)
}
```

Example:
```swift
struct Parameters : ParameterGroup {
    var w1 = Tensor<Float>(randomNormal: [784, 30])
    var b1 = Tensor<Float>(zeros: [30])
    var w2 = Tensor<Float>(randomNormal: [30, 10])
    var b2 = Tensor<Float>(zeros: [10])
}

var params = Parameters()
var optimizer = SGD()
for (x, ŷ) in dataset {
    let grads = gradient(at: params) { params in
        let h1 = tanh(matmul(x, params.w1) + params.b1)
        let y = sigmoid(matmul(h1, params.w2) + params.b2)
        let loss = (y - ŷ).squared().mean()
        print("Loss is \(loss)")
        return loss
    }
    optimizer.fit(&params, withGradients: grads)
}
```

2. Preserving original result

Since the trailing closure as an argument to `gradient(wrt:in:)`, the forward
computation is just as customizable as within operator-overloading AD systems.
Users can do whatever they want to intermediate values or the result in the
primal computation.

That said, we would like to provide a way to have the differentiation API return
the original result directly.

3. Delayed execution of derivative or gradient

```swift
/// Computes the gradient of the trailing closure at `x`.
func pushforward<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (T.TangentVector) -> R.TangentVector {
    return { direction in
        directionalDerivative(at: x, along: direction, in: body)
    }
}

/// Computes the gradient of the trailing closure at `x`.
func valueAndPullback<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (value: T, pullback: (R.CotangentVector) -> T.CotangentVector) {
    return { direction in
        gradient(at: x, in: body)
    }
}
```

## Applications

## Conclusion
