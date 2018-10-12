First-Class Automatic Differentiation in Swift: The Manifesto
=============================================================

* Author: [Richard Wei](https://github.com/rxwei)
* Date: October 2018

This manifesto is written for both the machine learning community and the Swift
programming language design community.

Table of Contents
-----------------
- [Introduction](#Introduction)
- [Brief History](#Brief-History)
- [Why?](#Why?)
- [Vision](#Vision)
- [System Design](#System-Design)
- [Conclusions](#Conclusions)
- [Acknowledgements](#Acknowledgements)

Introduction
------------

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

We aim to provide best-in-class AD, including the best optimizations, best error
messages in failure cases, and the most flexibility and expressivity. To achieve
this, we built support for AD right into the Swift compiler. Additionally, since
AD is important to the broader scientific and numerical computing communities,
we decided to build AD as a generic feature that is completely orthogonal to the
TensorFlow support - the TensorFlow Swift library computes gradients using the
AD features of the Swift language itself.

Brief History
-------------

Automatic differentiation has been a research topic in scientific computing and
high-performance computing for nearly half a century. Traditional tools such as
[OpenAD](http://www.mcs.anl.gov/OpenAD/),
[TAPENADE](http://tapenade.inria.fr:8080/tapenade/index.jsp) and
[ADIFOR](http://www.mcs.anl.gov/research/projects/adifor/) are tools that
transform existing source code. There are many advanced techniques that improved
the performance of derivatives written in FORTRAN, but these tools have not
gained wide adoption in the machine learning community. More recent AD systems
like [Stalin∇](https://github.com/Functional-AutoDiff/STALINGRAD) (pronounced
Stalingrad, available in Scheme),
[DiffSharp](http://diffsharp.github.io/DiffSharp/) (available in F#), and
[ad](https://hackage.haskell.org/package/ad) (available in Haskell) achieved
good usability by integrating the differential operator into the language, and
are equipped with a complete set of AD features (such as forward/reverse, nested
AD, Hessians, Jacobians, directional derivatives and checkpointing). They
combine AD closely with functional programming languages.

Researchers in the machine learning community have built many library
implementations of AD in Python and C++, including
[Autograd](https://github.com/HIPS/autograd),
[TensorFlow](http://tensorflow.org/), [Pytorch](http://pytorch.org/), etc. As
Automatic Differentiation is an integral part of any machine learning framework,
traditional designs and implementations of AD have some limitations. Here are
four kinds of existing approaches to automatic differentiation: 

| Approach to Automatic Differentiation               | Strength               | Limitation   |
| --------------------------------------------------- | ---------------------- | ------------ |
| Tracing + interpreting operators                    | Minimal implementation |              |
| Tracing + transforming traced program               |                        |              |
| Metaprogramming + interpreting operators            |                        |              |
| Metaprogramming + transforming program              |                        |              |


Some of these libraries are implemented as a transformation on a standalone DSL
(a graph) with a closed set of operators. Others are implemented using operator
overloading directly on a subset of the source language. Although these
libraries have gained wide adoption, the ones that leverage ahead-of-time AD do
not expose an easy-to-use programming model, and the ones that have a friendlier
programming model lack static analysis to perform more optimized AD.

Two recent projects ([Tangent](https://github.com/google/tangent) and
[Myia](https://github.com/mila-udem/myia)) based their AD upon source code
transformation (SCT), a technique that was common in advanced AD systems before
the deep learning era such as
[Stalin∇](https://github.com/Functional-AutoDiff/STALINGRAD). Both tools parse a
Python subset into ASTs and transform a function to its derivatives either in
AST or in a functional IR. These two projects fall into a category in deep
learning tools that was previously underexplored: ahead-of-time
differentiation + "model as code", as shown in the following diagram (cite:
[Tangent](https://github.com/google/tangent)). While these tools are pushing the
boundaries of Python, other research projects like [DLVM](http://dlvm.org/)
experimented with SCT AD directly on a compiler IR that's analogous to the
[Swift Intermediate
Language](https://github.com/apple/swift/blob/master/docs/SIL.rst) (SIL).

Most existing AD implementations work on a graph representation of a functional
tensor program, and many of them have limited expressivity and extensibility.
Frameworks based on a define-by-run programming model (to support dynamic
computation graphs) often lack the ability to perform full-program static
analysis and optimizations, and make it hard to diagnose errors and target
hardware accelerators ahead of time.

<p align="center">
  <img src="images/AutomaticDifferentiation-Approaches.png?raw=true"
       alt="Automatic differentiation approaches."/>
</p>

The horizontal axis of this diagram may remind people of the trade-offs between
[eager execution](https://www.tensorflow.org/guide/eager) and [computation graph
building](https://www.tensorflow.org/guide/graphs) in machine learning
frameworks: In eager execution, the model is a subset of user code. In graph
mode, the model is a data structure representing some code in a mini-language.
The [Graph Program
Extraction](https://github.com/tensorflow/swift/blob/master/docs/GraphProgramExtraction.md)
technique combines the best of both worlds by reducing graphs to an
implementation detail managed by the compiler. The vertical axis in the diagram
adds a second dimension, Automatic Differentiation, where Swift achieves exactly
the same by making AD a core feature of the language and the compiler.


Why?
----

Swift is a new programming language in the machine learning space, and it has a
special taste. Recently, the [Swift for
TensorFlow](https://github.com/tensorflow/swift) project brought the full power
of a machine learning framework into the Swift programming language, and the
machine learning community began to envision a world where mathematical
computation gains most first-class language support. Many articles have written
about Swift's potential impact in the machine learning space as a better
general-purpose language than Python that can serve their needs better, e.g. [Why
data scientists should start learning
Swift](https://heartbeat.fritz.ai/why-data-scientists-should-start-learning-swift-66c3643e0d0d).


During our development, we've prototyped two major extensions to the Swift
compiler in the ['tensorflow'
branch](https://github.com/apple/swift/tree/tensorflow).

- [Automatic
  Differentiation](https://github.com/tensorflow/swift/blob/master/docs/AutomaticDifferentiation.md):
  An extension to the syntax, type system, standard library, compiler
  transformations and ABI that makes Swift a first-class differentiable
  programming language.
- [Graph Program Extraction](https://github.com/tensorflow/swift/blob/master/docs/GraphProgramExtraction.md):
  An extension to the compiler that efficiently compiles control flow and data flow over
  TensorFlow's `Tensor` type and make the language talk to TensorFlow natively.

While Graph Program Extraction is specific to TensorFlow support, Automatic
Differentiation is completely independent of TensorFlow and has been designed as
a fully general language feature.


Vision
------

We believe that Swift will be a great platform for numerics and machine
learning.

We believe machine learning is so important today that it deserves first-class
language capabilities. Such capability need not be specialized for machine
learning uses cases, but will be driven by practical requirements nad 

**Swift will be world's first statically typed, general-purpose differentiable programming
language.**

### Differential Operators and Differentiation APIs

In basic calculus, differentiating a function of type ℝ → ℝ produces a
function that maps a point onto its slope, having type, and the derivative
function also has type ℝ → ℝ. In Swift terms, differentiating a function
`(Float) -> Float` produces `(Float) -> Float`, and differentiating a function
`(Float, Float) -> Float` produces `(Float, Float) -> (Float, Float)`.

In vector calculus, differentiation of a function $ℝ^n \rightarrow ℝ^m$ has many forms,
because there are multiple inputs and multiple outputs. A full evaluation of
derivatives of each output at each input will result into the following matrix,
called a [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

<p align="center">
  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/74e93aa903c2695e45770030453eb77224104ee4"
       alt="Automatic differentiation approaches."/>
</p>

Calculating the Jacobian of a function is very computationally expensive. In
practice, we often care more about two kinds of byproducts of Jacobian
calculation: the vector-Jacobian products (a.k.a. gradient) and the vector-Jacobian
products (a.k.a. directional derivatives). In these terms, "vector" refers to a
forward-propagated derivative that's to be chained with the function's
[differential](https://en.wikipedia.org/wiki/Pushforward_(differential)), and a
back-propagated gradient that's to be chained with the function's
[pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)),
respectively.

The most important aspect of a first-class automatic differentiation
system is the direct integration of differential operators in the language.

### Differentiation Styles (Functional vs. Imperative)

In common AD systems for machine learning, there are two differentiation styles
available: functional and imperative.

|            | Syntax | Meaning |
|------------|--------|-------------|
| Functional | `let 𝝯f = #gradient(f)`<br/>`𝝯f(x)` | Differentiating a function |
| Imperative | `y = f(x)`<br/>`gradient(of: y, wrt: x)` | Differentiating code traced through data flow |

Without getting into implementation details, functional-style AD is transforming
one function to another, producing a function that takes original
arguments and returns the partial derivatives evaluated at each argument.
Imperative-style AD, on the other hand, is a value-value transformation. In mathematics, we use both notations, i.e. both 


### High-Level Machine Learning APIs

System Design
-------------

### Differentiable Types

Differentiation is an operation defined on functions over mathematical objects.
In machine learning, we often differentiate functions over mathematical values.

#### A revised [`Numeric`](https://developer.apple.com/documentation/swift/numeric) protocol

The Numeric protocol today refines
[`ExpressibleByIntegerLiteral`](https://developer.apple.com/documentation/swift/expressiblebyintegerliteral).
This makes sense for scalars, but is not compatible with vector data structures
due to its 

The Swift team is working towards generalizing the numeric protocols so that
they will work well with vector libraries. On the Swift forum, I have discussed
the [fundamental blocker for vector types to conform to the existing `Numeric`
protocol](https://forums.swift.org/t/should-numeric-not-refine-expressiblebyintegerliteral).
[Steve Canon](https://forums.swift.org/u/scanon/summary) from the Swift
community stated that he may propose a weakened protocol for both `Numeric` and
a vector protocol to refine, and it's likely going to represent a
[rng](https://en.wikipedia.org/wiki/Rng_(algebra)) and be called `Arithmetic`.
In order to achieve a sensible design, we will assume this protocol already
exists so that the vector-related protocols we introduce can refine `Arithmetic`
to get properties of a rng (`0` and `+`).

```swift
public protocol Arithmetic : Equatable {
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

The existing `Numeric` will be changed to refine (inherit from) `Arithmetic`.

```swift
public protocol Numeric : Arithmetic, ExpressibleByIntegerLiteral {
    associatedtype Magnitude : Comparable, Numeric
    init?<T>(exactly source: T) where T : BinaryInteger
    var magnitude: Magnitude { get }
}
```

#### The `VectorNumeric` protocol

After we make `Arithmetic` suitable for machine learning use cases, we can
define a protocol that generalizes vectors. Mathematically, a vector space is a
rng -- a ring without a multiplicative identity. We represent vector spaces
through the `VectorNumeric` protocol as follows. `ScalarElement` is the type of
the elements of this vector space -- the field which the vector space is over.
`Dimensionality` is the dimensionality of this vector space, which is
customizable. The initializer takes a value of the `ScalarElement` type and a
`Dimensionality` and returns a vector of the specified dimensionality.

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

  /// The dimensionality of this vector.
  var dimensionality: Dimensionality { get }

  /// Returns the scalar product of the vector.
  static func * (scale: ScalarElement, value: Self) -> Self
}
```

#### The `Differentiable` protocol

In order for values of a type to be differentiable, the type has to satisfy the
following requirements:

1. Derivatives form a vector space - it has `0`, `1`, and addition (`+`).
2. Derivatives can be applied to perform
   [optimizations](https://en.wikipedia.org/wiki/Mathematical_optimization).

In mathematics, this is usually generalized by two kinds of 

Here's the finished `Differentiable` protocol.

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
    /// vector. In Riemannian geometry (mathematics), this is usually equivalent
    /// to retraction or exponential map.
    func moved(toward direction: TangentVector) -> Self
  
    /// Convert a cotangent vector to its corresponding tangent vector.
    func tangentVector(from cotangent: CotangentVector) -> TangentVector
}
```

Why do we need a customizable `TangentVector` and a custom `CotangentVector`?
While most machine learning frameworks model derivatives as 

When the tangent vector of a differentiable manifold is equal to its cotangent
vector, we can simply provide a default implementation of
`tangentVector(from:)`, which is just the identity function.

```swift
public extension Differentiable where TangentVector == CotangentVector {
    func tangentVector(from cotangent: CotangentVector) -> TangentVector {
        return cotangent
    }
}
```

When a differentiable manifold is a vector space, and when the tangent space
equals manifold itself, its exponential ma

```swift
public extension Differentiable
    where Self : VectorNumeric, TangentVector == Self {
    func moved(toward direction: TangentVector) -> Self {
        return self + direction
    }
}
```

When a differentiable manifold is a vector space, it's tangent space is usually
itself. In these cases, we simply define `moved(toward:)` as vector addition.

There are three important use cases of such a generalization.

1. Customizable weight type
   
   Orthogonal weight matrixes have shown advantages in neural network training
   [[1]](https://arxiv.org/abs/1702.00071)
   [[2]](https://arxiv.org/abs/1709.06079). When differentiating through these
   networks, gradients with respect to weights will no long stay orthogonal -
   instead, they are skew-symmetric matrices. While we can represent both
   orthogonal matrices and skew-symmetric matrices as values of a `Matrix` or
   `Tensor` type and programmatically ensure its orthogonality, some researchers
   have been seeking a way to represent this natively in the type ystem of a
   programming language and still have AD produce the correct derivative.

2. Quantized training

   Quantization techniques store and calculate numbers in more compact formats,
   i.e. a fixed-point data type. Conceptually, a quantized tensor for a
   real-valued `Tensor` can be defined as the following struct:

   ```swift
   struct QuantizedTensor<OriginalScalar, QuantizedScalar>
       where OriginalScalar: AccelerableByTensorFlow & FloatingPoint,
             QuantizedScalar: AccelerableByTensorFlow & FixedWidthInteger {
       var data: Tensor<Int8>
       var range: Range<Float>
       var scale: Float
       var zeroPoint: Int32
   }
   ```

   We can think of a scenario where the developer defines a neural network as a
   function whose parameters are of type `QuantizedFloat`. When training
   parameters to this neural network, the function needs to be

   ```swift
   // `QuantizedTensor` is a vector space.
   extension QuantizedTensor : VectorNumeric {
       typealias ScalarElement = OriginalScalar
       // Implementations of `+` and `*` are omitted.
   }

   // `QuantizedTensor` is a differentiable manifold.
   extension QuantizedTensor : Differentiable {
       typealias TangentVector = Tensor<OriginalScalar>
       typealias CotangentVector = Tensor<OriginalScalar>
   }
   ```

   One would expect to differentiate a function of type `(QuantizedTensor<Float,
   Int8>) -> U` to train parameters. However, we never want gradients at each
   parameter to be a quantized number - instead, the range and zero point are
   often recomputed for each parameter update.

3. Generic optimizers

   Optimization problems in machine learning is generalized by 

   This, in fact, has been exactly what people have been doing when writing machine
   learning optimizers, but the conversion has often been implicit because in most
   cases the tangent space is equal to the cotangent space in a vector space
   scenario. With this generalization, people will be able to write general
   optimizers that is both practically general and mathematically correct.

### Differential Registration

We are aiming for an open and extensible system, so we made the compiler
agnostic of the actual operations - it does not have special knowledge of
numeric standard library functions or distinguish between primitive operators
and other functions. We recursively determine a function's differentiability
based on:

- Whether a function is differen

#### The `@differentiable` attribute

We introduce a declaration attribute `@differentiable` to the language.

```ebnf
differentiation-mode = 'forward' | 'reverse'
differentiability = differentiation-mode | 'bidirectional' | 'linear' | 'constant'
differentiable-attribute = '@differentiable' '(' differentiability ')'
declaration-attribute = differentiable-attribute
```

### Raw Differential Operators

Now that we have talked about differentiable types and differential
registration, it's time to move to the core differentiation API.

```ebnf
forward-differential-operator = '#derivative'
reverse-differential-operator = '#gradient'
differential-operator = forward-differential-operator | reverse-differential-operator
autodiff-argument-index-specifier = '.' integer-literal
autodiff-expression = 
    differential-operator '(' expression [ ',' 'wrt' ':' autodiff-argument-index-specifier ] ')'
expression = autodiff-expression
```

Example:
```swift
func f(_ x: Float, _ y: Float) -> Float {
    return sin(x + y)
}
#derivative(f) // (Float, Float) -> (Float?) -> (Float, (Float, Float))
               //  ^~~~~~~~~~~~      ^~~~~      ^~~~~   ^~~~~~~~~~~~
               //  original args     seed       result  derivative
#gradient(f) // (Float, Float) -> (Float, (Float?) -> (Float, Float))
             //  ^~~~~~~~~~~~      ^~~~~   ^~~~~      ^~~~~~~~~~~~
             //  original args     result  seed       gradient
```

### Generalized Differentiability

Automatic differentiation relies on the source code of a function to be able to
differentiate it. Differential operators like `#gradient` trigger the
differentiation of a function, and the differentiability of the function is
determined as differentiation goes. This simple system works perfectly when
differentiating concrete functions defined in a local module, but does not allow
differentiation of opaque function values or methods required by protocols.
While the former is not a struct requirement in machine learning systems, the
latter fundamentally obstructs composable, protocol-oriented machine learning
APIs. This document describes a new formalization of differentiability in
Swift's type system, including an `@autodiff` function type attribute, an
extension to functions' layout, and new syntax for selecting differentiable
arguments. 

### Differentiation APIs

Previously, we introduced keywords `#derivative` and `#gradient` that represent
differential operators.

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
    return #chainableDerivative(body)(x)(direction)
}
```

```
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

#### Delayed execution of derivative or gradient ([differentials](https://en.wikipedia.org/wiki/Pushforward_(differential)) and [pullbacks](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)))

In some reinforcement learning (RL) tasks, it is often required to compute the
forward pass of a neural network, hand it over to other functions, wait for
the adjoint to be back-propagated, and then compute the backward computation.

```swift
/// Computes the differential of `body` at `x`.
func differential<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (T.TangentVector) -> (originalResult: T, derivative: R.TangentVector) {
    return #gradient(body)(x)
}

/// Computes the original value of `body(x)` and the pullback (chainable gradient)
/// at `x`.
func pullback<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (originalResult: T, pullback: (R.CotangentVector) -> T.CotangentVector) {
    return #valueAndPullback(body)(x)
}
```

Examples:

1. Chain directional derivatives freely using differentials.

    ```swift
    let x = 0.5
    let df = differential(at: x) { x in
        sin(cos(x))
    }
    df(1) // (f(x), df/dx)
    df(#derivative(log)(t)) // (f(x), df/dt)
    df(derivative(at: t, in: log)) // (f(x), df/dt)
    ```

2. Chain gradients freely using pullbacks.

    ```swift
    let x = 0.5
    let (y, df) = valueAndPullback(at: x) { x in
        cos(sin(x))
    }
    
    df(1) // dy/dx
    df(#gradient(log)(t)) // dy/dt
    df(gradient(at: t, in: log)) // dy/dt
    ```

#### Customizable differentiation

Some machine learning models require manipulating the gradient with respect to
certain values, e.g. gradient clipping.
[Tangent](https://github.com/google/tangent) provides such a feature as a syntax
extension in Python. Recurrent neural networks often suffer from the "exploding
gradient" problem, and a typical solution is to force the gradient of an RNN to
not exceed a certain value by performing gradient clipping.

```swift
func prediction(for input: Tensor<Float>) -> Float {
    var prediction = input
    for _ in 0...5 {
        // Clip gradient.
        prediction = prediction.withCustomizedGradient { grad in
            max(min(grad, 1), -1)
        }
        prediction = lstm.prediction(for: input)
    }
    return prediction
}
```

The `withCustomizedGradient` API looks like a compiler-known function which makes
Swift run customized gradient computation. However, because of the
generality of the [differential registration](#differential-registration)
machanism, this API can be defined entirely as a Swift function with no special
support from the compiler. Here's the implementation of the gradient
customization API.

```swift
public extension Differentiable {
    @differentiable(reverse, wrt: self, adjoint: adjointCustomizingGradient)
    func withCustomizedGradient<R>(
        using body: @nondiff (CotangentVector?) -> CotangentVector?
    ) -> Self {
        return self
    }

    internal func adjointCustomizingGradient(
        body: (CotangentVector?) -> CotangentVector?,
        originalResult: Self,
        adjoint: CotangentVector?
    ) -> CotangentVector? {
        return body(adjoint)
    }
}
```

The derivative customization API is just as simple.

```swift
public extension Differentiable {
    @differentiable(forward, wrt: self, tangent: tangentCustomizingGradient)
    func withCustomizedDerivative(
        using body: @nondiff (TangentVector?) -> TangentVector?
    ) -> Self {
        return value
    }

    internal func adjointCustomizingGradient(
        body: (TangentVector?) -> TangentVector?,
        tangent: Self?
    ) -> TangentVector? {
        return body(tangent)
    }
}
```

This API supports all general gradient manipulation tasks in machine learning
optimization. For example, [stop
gradient](https://www.tensorflow.org/api_docs/python/tf/stop_gradient) can be
implemented simply by `break`ing from the loop.
```swift
var prediction = input
for _ in 0...5 {
    // Stop gradient when necessary.
    var shouldStop = false
    prediction = prediction.withCustomizedGradient { grad in
        if grad < lowerBound {
            shouldStop = true
        }
        return grad
    }
    if shouldStop {
        break
    }
    prediction = lstm.prediction(for: input)
}
```

Setting a mutable flag is not the most user-friendly way. We can create an API
that wraps `withCustomizedDerivative(using:)` and returns a `Bool`, so that later code
can decide whether to `break` from the loop based on the return value from that
API.

Conclusions
-----------

Acknowledgements
----------------

Special thanks to Dan Zheng, Alex Wiltschko, Bart van Merriënboer, Dougal
Maclaurin, Matthew Johnson, Gordon Plotkin, Tim Harley, Malcolm Reynolds, Marc
Rasi, and Chris Lattner for their input to the design of this powerful language
feature.
