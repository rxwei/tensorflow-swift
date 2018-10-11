# The Automatic Differentiation Manifesto

* Author: [Richard Wei](https://github.com/rxwei)
* Date: October 2018

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

#### Revised `Numeric` protocol

The Swift team will introduce that mathematically represents a
[rng](https://en.wikipedia.org/wiki/Rng_(algebra)).

```swift
protocol Arithmetic : Equatable {
    var zero: Self { get }
    prefix static func + (x: Self) -> Self
    static func + (lhs: Self, rhs: Self) -> Self
    static func += (lhs: inout Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func -= (lhs: inout Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func *= (lhs: inout Self, rhs: Self) -> Self
}
```

```swift
protocol Numeric : Arithmetic, ExpressibleByIntegerLiteral {
    associatedtype Magnitude : Comparable, Numeric
    init?<T>(exactly source: T) where T : BinaryInteger
    var magnitude: Magnitude { get }
}
```


```swift
/// A type that represents an unranked vector space. Values of this type are
/// elements in this vector space and with a specific dimensionality.
public protocol VectorNumeric : Arithmetic {
  /// The type of scalars in the real vector space.
  associatedtype ScalarElement

  /// The type whose values specifies the dimensionality of an object in the
  /// real vector space.
  associatedtype Dimensionality

  /// Create a scalar in the real vector space that the type represents.
  ///
  /// - Parameter scalar: the scalar
  init(_ scalar: ScalarElement)

  /// Create an object in the real vector space with the specified
  /// dimensionality by repeatedly filling the object with the specified
  /// value.
  ///
  /// - Parameters:
  ///   - repeatedValue: the value repeat for the specified dimensionality
  ///   - dimensionality: the dimensionality
  init(repeating repeatedValue: ScalarElement, dimensionality: Dimensionality)

  static func * (scale: ScalarElement, value: Self) -> Self
}
```


#### The `Differentiable` protocol

Differentiable manifolds are awesome.

```swift
/// A type that mathematically represents a differentiable manifold whose
/// tangent spaces are finite-dimensional.
///
/// In automatic differentiation, differentiation will produce a Jacobian whose
/// elements are of `Tangent` type.
public protocol Differentiable {
  /// The tangent vector space of this differentiable manifold.
  associatedtype TangentVector : VectorNumeric
    where TangentVector.ScalarElement : FloatingPoint
  /// The cotangent space of this differentiable manifold.
  associatedtype CotangentVector : VectorNumeric
    where TangentVector.ScalarElement : FloatingPoint

  /// Returns `self` moved along the value space towards the given tangent
  /// vector. In Riemannian geometry (mathematics), this represents an
  /// exponential map.
  func moved(toward direction: TangentVector) -> Self
  
  /// Convert a cotangent vector to its corresponding tangent vector.
  func tangentVector(from cotangent: CotangentVector) -> TangentVector
}
```

When a differential manifold is a vector space, it's tangent space is usually
itself. In these cases, we can define `moved(toward:)` as addition.

```swift
public extension Differentiable
  where Self : VectorNumeric, TangentVector == Self {
  func moved(toward direction: TangentVector) -> Self {
    return self + direction
  }
}
```

```swift
public extension Differentiable where TangentVector == CotangentVector {
  func tangentVector(from cotangent: CotangentVector) -> TangentVector {
    return cotangent
  }
}
```

### Derivative Registration

#### The `@differentiable` attribute

We introduce a declaration attribute `@differentiable` to the language. This
attribute can be ap

```ebnf
differentiation-mode = 'forward' | 'reverse'
differentiability = differentiation-mode | 'bidirectional' | 'linear' | 'constant'
differentiable-attribute = '@differentiable' '(' differentiability ')'
declaration-attribute-list = differentiable-attribute declaration-attributes
```

```ebnf

```

### Raw Differential Operators

```ebnf
forward-differential-operator = '#derivative' | '#chainableDerivative'
reverse-differential-operator = '#gradient' | '#valueAndGradient' | '#chainableGradient'
differential-operator = forward-differential-operator | reverse-differential-operator
autodiff-expression = differential-operator '(' expression [ ',' 'wrt' ':'  ] ')'
expression = autodiff-expression
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

Mathematically, differentiability is a notion defined around functions.

### Differentiation API



#### Directional derivatives and gradients

```swift
/// Computes the derivative of the trailing closure at `x`.
func derivative<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(forward) (T) throws -> R
) rethrows -> R.TangentVector {
    return #derivative(body)(x)
}

/// Computes the directional derivative at `x` along the direction `u`.
func directionalDerivative<T : Differentiable, R : Differentiable>(
    at x: T, along direction: T.TangentVector, 
    in body: @autodiff(forward) (T) throws -> R
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

// Overloads for more arguments are omitted.
```

Application:
```swift
struct Parameters : ParameterGroup {
    var w1 = Tensor<Float>(randomNormal: [784, 30])
    var b1 = Tensor<Float>(zeros: [30])
    var w2 = Tensor<Float>(randomNormal: [30, 10])
    var b2 = Tensor<Float>(zeros: [10])
}

var params = Parameters()
var optimizer = StochasticGradientDescent()
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

#### Preserving original result

Since the trailing closure as an argument to `gradient(at:in:)`, the forward
computation is just as customizable as within operator-overloading AD systems.
Users can do whatever they want to intermediate values or the result in the
primal computation.

That said, we would like to provide a way to have the differentiation API return
the original result directly.

#### Delayed execution of derivative or gradient

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
    return #valueAndPullback(body)(x)
}
```

## Applications

## Conclusion
