First-Class Automatic Differentiation in Swift: The Manifesto
=============================================================

* Author: [Richard Wei](https://github.com/rxwei)
* Date: October 2018

This manifesto is written for both the machine learning community and the Swift
programming language design community.

Table of Contents
-----------------
- [Introduction](#Introduction)
- [Why does Swift need AD?](#Why-1)
- [Why make AD first-class?](#Why-2)
- [Vision](#Vision)
- [Part 1: Differentiable Types](#Part-1-Differentiable-Types)
- [Part 2: Primitive Registration](#Part-2-Primitive-Registration)
- [Part 3: Basic Differentiation](#Part-3-Basic-Differentiation)
- [Part 4: Generalized Differentiability](#Part-4-Generalized-Differentiability)
- [Part 5: True Differential Operators](#Part-5-True-Differential-Operators)
- [Part 6: Generalized Types for Differentiation](#Part-6-Generalized-Types-for-Differentiation)
- [Part 7: Customizable Differentiation](#Part-7-Customizable-Differentiation)
- [Applications](#Applications)
- [Future Directions](#Future-Directions)
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
learning methods.

We aim to provide best-in-class AD, including the best optimizations, best error
messages in failure cases, and the most flexibility and expressivity. To achieve
this, we built support for AD right into the Swift compiler. Additionally, since
AD is important to the broader scientific and numerical computing communities,
we decided to build AD as a generic feature that is completely orthogonal to the
TensorFlow support - the TensorFlow Swift library computes gradients using the
AD features of the Swift language itself.

What is AD?
-----------

### Basic Calculus

In basic calculus, differentiating a function of type `ℝ → ℝ` produces a function
ℝ → ℝ that maps points onto their corresponding slopes.

<p align="center">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9315f1516ee5847107808697e43693d91abfc6e8"
</p>

In the context of Swift, differentiating a function `(Float) -> Float` produces
`(Float) -> Float`. Functions with multiple arguments, such as `(Float, Float)
-> Float`, can be thought of as a function whose input domain is a product of
those arguments types, i.e. `(ℝ ⨯ ℝ) → ℝ`, so the derivative of such a function
has type `(Float, Float) -> (Float, Float)`. According to this typing rule, the
differential operator ![](http://latex.codecogs.com/gif.latex?\dfrac{d}{dx}) can
be declared as a higher-order function, overloaded for each number of arguments
because a Swift function's argument list is not formally modeled as a tuple.

```swift
func 𝒟<T : FloatingPoint>(_ f: (T) -> T) -> (T) -> T
func 𝒟<T : FloatingPoint>(_ f: (T, T) -> T) -> (T) -> (T, T)
func 𝒟<T : FloatingPoint>(_ f: (T, T, T) -> T) -> (T) -> (T, T, T)
...
```

```swift
func f(_ x: Double, _ y: Double) -> Double {
    return tanh(x + y)
}
𝒟(f) // (Double, Double) -> (Double, Double)
```

### Vectors and Jacobians

In numerical computing, users often write code that operate on high-dimensional
mathematical objects. The basic typing rules that we defined on real scalars (ℝ)
can be generalized for
[module](https://en.wikipedia.org/wiki/Module_(mathematics))-like types such as
vectors with extra consideration for dimensionality. In vector calculus, the
differentiation of a function `f: ℝⁿ → ℝᵐ` is defined per scalar because there
are multiple inputs and multiple outputs. Full differentiation of vector
function `f` will result in a matrix, each of whose entries is a function that
computes the partial derivatives of an output scalar with respect to an input
scalar. This matrix is called a
[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant). In
this definition, the Jacobian matrix has type: `J: (ℝ → ℝ)ᵐⁿ`. For simplicity,
we will model it as a function that maps vectors to real-valued matrices `J: ℝⁿ
→ ℝᵐⁿ`.

<p align="center">
  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/74e93aa903c2695e45770030453eb77224104ee4"
       alt="Automatic differentiation approaches."/>
</p>

While it is challenging to define this function with full type safety in Swift
because dimensionality cannot be generic parameters yet, we can define a
differential operator as the following, specialized on dimensionality.

```swift
func 𝒟<T>(_ f: (Vector2<T>) -> Vector3<T>) -> (Vector2<T>) -> Matrix3x2<T>
    where T : FloatingPoint
```

Calculating the Jacobian of a function is very computationally expensive, and is
often unnecessary in gradient-based optimization methods. In practice, we care
more about two byproducts of Jacobian calculation that are significantly easier
to compute than the Jacobian itself: the vector-Jacobian products and the
Jacobian-vector products. In these terms, "vector" refers to a vector of partial
derivatives that are to be chained with the Jacobian by left-multiplication or
right-multiplication. As we explain chaining next, we discuss how Automatic
Differentiation comes in the picture.

### Gradient and Reverse-Mode AD

When we let a [one-hot](https://en.wikipedia.org/wiki/One-hot) row vector
`vⁱ: ℝⁿ = onehot(i)` left-multiply a
Jacobian value matrix of type `ℝᵐⁿ`, we are selecting one row in the matrix,
which is exactly the [gradient](https://en.wikipedia.org/wiki/Gradient) of
![](http://latex.codecogs.com/gif.latex?f_i) evaluated at
![](http://latex.codecogs.com/gif.latex?x), i.e.
![](http://latex.codecogs.com/gif.latex?\nabla{f_i}(\mathbf{x})).

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\nabla{f_i}(\mathbf{x})=\mathbf{v}^i\mathbf{J_f}(\mathbf{x})=\bigg[\dfrac{\partial{f_i}(\mathbf{x})}{\partial&space;x_0}&space;\&space;\cdots&space;\&space;\dfrac{\partial{f_i}(\mathbf{x})}{\partial{x_n}}\bigg]" title="\nabla{f_i}(\mathbf{x})=\mathbf{v}^i\mathbf{J_f}(\mathbf{x})=\bigg[\dfrac{\partial{f_i}(\mathbf{x})}{\partial x_0} \ \cdots \ \dfrac{\partial{f_i}(\mathbf{x})}{\partial{x_n}}\bigg]" />
</p>

When this vector represents the gradient of another function `g: ℝᵐ → ℝ` at
![](http://latex.codecogs.com/gif.latex?\mathbf{f}(\mathbf{x})), namely
![](http://latex.codecogs.com/gif.latex?\partial{g(\mathbf{y})}/\partial{f_i(\mathbf{x})}),
then the vector-Jacobian products will represent
![](http://latex.codecogs.com/gif.latex?\partial{g(\mathbf{y})}/\partial{\mathbf{x}}).
The function that takes a vector and left-multiplies it with the Jacobian is
also called a
[pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)). We
can define this function in Swift as a higher-order function shown below. The
body of this function can be defined in terms of `𝒟`, the differential operator
that returns a Jacobian.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dfrac{\partial&space;g(\mathbf{f}(\mathbf{x}))}{\partial&space;\mathbf{x}}=\dfrac{\partial&space;g}{\partial&space;\mathbf{f}(\mathbf{x})}\mathbf{J_f}(\mathbf{x})&space;=&space;\bigg[&space;\dfrac{\partial&space;g(\mathbf{x})}{\partial&space;x_0}&space;\&space;\cdots&space;\&space;\dfrac{\partial&space;g(\mathbf{x})}{\partial&space;x_n}&space;\bigg]" title="\dfrac{\partial g(\mathbf{f}(\mathbf{x}))}{\partial \mathbf{x}}=\dfrac{\partial g}{\partial \mathbf{f}(\mathbf{x})}\mathbf{J_f}(\mathbf{x}) = \bigg[ \dfrac{\partial g(\mathbf{x})}{\partial x_0} \ \cdots \ \dfrac{\partial g(\mathbf{x})}{\partial x_n} \bigg]" />
</p>

```swift
func pullback<T: FloatingPoint>(
    of f: (Vector2<T>) -> Vector3<T>,
    at x: Vector2<T>
) -> (Vector2<T>) -> Vector2<T>
    return { adjoint in matmul(adjoint, 𝒟(f)(x)) }
}
```

However, when computing gradients or general vector-Jacobian products, we do not
need to compute the Jacobian at all: **Automatic Differentiation is here to
help.**

*The chain rule of differentiation* is defined as follows.

<!-- Chain rule -->

The chain rule can be interpreted in either left-associative order, i.e.
accumulating each function's partial derivatives from the final output,
eventiually reaching each input.

### Directional Derivatives and Forward-Mode AD

Similarly, when we let a column vector `v: ℝⁿ¹` right-multiply a Jacobian value
matrix of type `ℝᵐⁿ`, the result is a vector whose elements are exactly the
[directional derivatives](https://en.wikipedia.org/wiki/Directional_derivative)
of each ![](http://latex.codecogs.com/gif.latex?f_i) evaluated at
![](http://latex.codecogs.com/gif.latex?\mathbf{x}) in direction ![](http://latex.codecogs.com/gif.latex?\mathbf{v}).

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\nabla_\mathbf{v}\mathbf{f}(\mathbf{x})=\mathbf{J_f}(\mathbf{x})\mathbf{v}=\bigg[\nabla_\mathbf{v}{f_0}(\mathbf{x})\&space;\cdots\&space;\nabla_\mathbf{v}{f_m}(\mathbf{x})\bigg]" title="\nabla_\mathbf{v}\mathbf{f}(\mathbf{x})=\mathbf{J_f}(\mathbf{x})\mathbf{v}=\bigg[\nabla_\mathbf{v}{f_0}(\mathbf{x})\ \cdots\ \nabla_\mathbf{v}{f_m}(\mathbf{x})\bigg]" />
</p>

The function that takes a vector and right-multiplies the Jacobian value matrix
is called a
[differential](https://en.wikipedia.org/wiki/Pushforward_(differential)), and it
can also be defined in Swift as a higher-order function in terms of `𝒟`.

```swift
func differential<T: FloatingPoint>(
    of f: (Vector2<T>) -> Vector3<T>,
    at x: Vector2<T>
) -> (Vector3<T>) -> Vector3<T> {
    return { tangent in matmul(𝒟(f)(x), tangent) }
}
```

Just like vector-Jacobian products, Jacobian-vector products are easy to compute
using Automatic Differentiation. By simply applying the chain rule of
differentiation from an input, we will accumulate each function's partial
derivatives and reach each output.

AD has a rich background. For an in-depth introduction, here's some great
documentation: [Introduction to Automatic
Differentiation](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/)
and [Automatic Differentiation in Machine Learning: a
Survey](https://arxiv.org/abs/1502.05767).

Why does Swift need AD?
-----------------------

Swift is a new programming language in the machine learning space. Recently, the
[Swift for
TensorFlow](https://github.com/tensorflow/swift) project brought the full power
of a machine learning framework into the Swift programming language, and the
machine learning community began to envision a world where mathematical
computation gains most first-class language support. Many articles have written
about Swift's potential impact in the machine learning space as a better
general-purpose language than Python that can serve their needs better, e.g.
[Why data scientists should start learning
Swift](https://heartbeat.fritz.ai/why-data-scientists-should-start-learning-swift-66c3643e0d0d).

Numerical computing has a very different set of requirements than application
development and system development. Because of Python's extensibility and lack
of strong static types, a complete ecosystem has been developed to suit
developers' need, such as various automatic differentiation libraries.

One of the most important building blocks in mathematical computing is the
ability to differentiate math code, in one form or another.

During our development, we've prototyped two major extensions to the Swift
compiler in the ['tensorflow'
branch](https://github.com/apple/swift/tree/tensorflow).

<!-- - [Automatic
  Differentiation](https://github.com/tensorflow/swift/blob/master/docs/AutomaticDifferentiation.md):
  An extension to the syntax, type system, standard library, compiler
  transformations and ABI that makes Swift a first-class differentiable
  programming language.
- [Graph Program Extraction](https://github.com/tensorflow/swift/blob/master/docs/GraphProgramExtraction.md):
  An extension to the compiler that efficiently compiles control flow and data flow over
  TensorFlow's `Tensor` type and make the language talk to TensorFlow natively. -->

While Graph Program Extraction is specific to TensorFlow support, Automatic
Differentiation is completely independent of TensorFlow and has been designed as
a fully general language feature.

Why make AD first-class?
---------------------------

Automatic Differentiation has been a research topic in scientific computing and
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
[TensorFlow](http://tensorflow.org/), [Pytorch](http://pytorch.org/), etc.

As Automatic Differentiation is an integral part of any machine learning
framework, traditional designs and implementations of AD have some limitations.
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


Existing frameworks implement the following four styles of automatic
differentiation.

1. Trace program execution and interpret overloaded primitive operators

    The benefit of this is that

2. Trace program execution and transform the program

3. Metaprogram and interpret overloaded operators

4. Metaprogram and transform the program


Vision
------

**Swift will be world's first general-purpose differentiable programming
language.**

### Differential Operators and Differentiation APIs

The most important aspect of a first-class automatic differentiation
system is the direct integration of differential operators in the language.

### Differentiation Styles (Functional vs. Imperative)

In common AD systems for machine learning, there are two differentiation styles
available: functional and imperative.

|            | Syntax | Meaning |
|------------|--------|-------------|
| Functional | `let 𝝯f = gradient(of: f)`<br/>`𝝯f(x)` | Differentiating a function |
| Imperative | `y = f(x)`<br/>`gradient(of: y, wrt: x)` | Differentiating code traced through data flow |

Without getting into implementation details, functional-style AD is transforming
one function to another, producing a function that takes original
arguments and returns the partial derivatives evaluated at each argument.
Imperative-style AD, on the other hand, is a value-value transformation. In mathematics, we use both notations, i.e. both


### High-Level Machine Learning APIs



Part 1: Differentiable Types
----------------------------

Swift is a general-purpose programming language. Therefore, not every function
is mathematically differentiable, and not every type represents a real vector
space to begin with. To make our system mathematically sound, we refine the
Swift standard library to form a basis for automatic differentiation.

The starting point of this refinement is the fundamental numeric protocols.
In this section, we talk about how we improve the `Numeric` protocol to support
the addition of vector types and protocols. Then, we introduce a protocol to
represent vector spaces as that would be a requirement for doing calculus.
Finally, we design a protocol specific to differentiation.

### Revising the [`Numeric`](https://developer.apple.com/documentation/swift/numeric) protocol

The Numeric protocol today refines
[`ExpressibleByIntegerLiteral`](https://developer.apple.com/documentation/swift/expressiblebyintegerliteral).
This makes sense for scalars, but is not compatible with vector data structures
because type-checking would fail on the scalar multiplication operator.

On the Swift forum, we have discussed the [fundamental blocker for vector types to conform to the existing `Numeric`
protocol](https://forums.swift.org/t/should-numeric-not-refine-expressiblebyintegerliteral).
The consensus was to introduce a weakening of the `Numeric` protocol to
represent the abstractions shared between scalars and vectors:
[rng](https://en.wikipedia.org/wiki/Rng_(algebra)).
The protocol will be called `Arithmetic`.

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

The existing `Numeric` will be changed to refine (inherit from) `Arithmetic`,
keeping all of its existing behavior.

```swift
public protocol Numeric : Arithmetic, ExpressibleByIntegerLiteral {
    associatedtype Magnitude : Comparable, Numeric
    init?<T>(exactly source: T) where T : BinaryInteger
    var magnitude: Magnitude { get }
}
```

### The `VectorNumeric` protocol

After we introduce the `Arithmetic` protocol, which makes the standard library
suitable for vector APIs and beyond,
we can define a protocol that generalizes vectors. Mathematically, a vector space is a
rng -- a ring without a multiplicative identity. We represent vector spaces
through the `VectorNumeric` protocol as follows. `ScalarElement` is the type of
the elements of this vector space -- the field which the vector space is over.
`Dimensionality` is the dimensionality of this vector space, which is
customizable. The initializer takes a value of the `ScalarElement` type and a
`Dimensionality` and returns a vector of the specified dimensionality.

```swift
/// A type that represents an unranked vector space. Values of this type are
/// elements in this vector space and with a specific dimensionality.
public protocol VectorNumeric: Arithmetic {
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

### The `Differentiable` protocol

Now we define a protocol that "activates" a type's differentiablity.
At a first glance, the conforming type must also be a `VectorNumeric` type.
So we make this protocol refine `VectorNumeric`. Since differentiation only
makes sense on real vectors, we add a constraint on the associated type
`ScalarElement` so that it conforms to `FloatingPoint`.

```swift
public protocol Differentiable : VectorNumeric where ScalarElement : FloatingPoint {
}
```

You may notice that `Differentiable` looks like a dummy protocol because it
doesn't have any requirements other than the ones inherited from
`VectorNumeric`. Although under the current assumptions we can completely omit
the `Differentiable` protocol and just have the AD system recognize
`VectorNumeric`-comforming types whose scalar elements comform to
`FloatingPoint`, we actually have theoretical and practical reasons to revise
the `Differentiable` protocol later on. So we keep `Differentiable` as a
separate protocol for now and build towards the final design at the end of this
document.

Part 2: Primitive Registration
------------------------------

We are aiming for an open and extensible system, so we made the compiler
agnostic of the actual operations - it does not have special knowledge of
numeric standard library functions or distinguish between primitive operators
and other functions. We recursively determine a function's differentiability
based on:

- whether a function has a primitive differentiability as specified in the
  standard or user-defined library, and

- whether a function's definition (type signature and body) is differentiable by
  applying the chain rule of differentiation.

As such we provide a syntactic way of specifying the differentiability of a
function, using either the function's linearity properties or a separate
function to specify the "tangent code" or "adjoint code" for the original
function.

### The `@differentiable` attribute

We introduce a declaration attribute `@differentiable` to Swift's syntax. The
full grammar of `@differentiable` is defined as follows:

```ebnf
differentiation-mode = 'forward' | 'reverse' | 'bidirectional'
differentiability = differentiation-mode  | 'linear' | 'constant'
differentiability-wrt-self = 'wrt' ':' 'self'
differentiation-order = 'once'
differentiation-tangent-specifier = 'tangent' ':' declaration-name
differentiation-adjoint-specifier = 'adjoint' ':' declaration-name
differentiable-attribute = '@differentiable'
    '(' differentiability
    [ ',' differentiability-wrt-self ]
    [ ',' differentiation-once ]
    [ ',' differentiation-tangent-specifier ]
    [ ',' differentiation-adjoint-specifier ]
    ')'
declaration-attribute = differentiable-attribute
```

#### Differentiation parameters

   Differentiation parameters are marked inline at each argument position in the
   function declaration. By default, every argument of the funtion is to be
   differentiated with-respect-to, unless marked as `@nondiff`.

   When a differentiable attribute is applied on a method, or the getter of a
   computed property in a type, the implicit `self` argument often needs to be
   differentiated with respect to. In order to make a function a primitive
   differentiable with respect to `self`, one can add a `wrt: self` to
   the `@differentiable` attribute.

#### Differentiability

There are five options for differentiablity:

1. Forward: `@differentiable(forward, tangent: ...)`

   This option says that the function is forward-mode differentiable.
   Forward-mode differentiation requires the "tangent code" (or tangent
   function) of this function, so that Swift knows how to compute the
   function's directional derivatives in the direction specified by the
   tangent vector that has been forward-propagated to the tangent function.

   The compiler will expect the identifier of the tangent function, with an
   expected type signature, to be specified later in the `tangent:` parameter
   in the attribute.

2. Reverse: `@differentiable(reverse, adjoint: ...)`

   This option says that the function is reverse-mode differentiable.
   Reverse-mode differentiation requires the "adjoint code" (or adjoint
   function) of this function, so that Swift knows how to compute the
   function's vector-Jacobian products, where the vector, or called "adjoint
   vector" has been back-propagated to the adjoint function.

   The compiler will expect the identifier of the adjoint function, with an
   expected type signature, to be specified later in the `adjoint:` parameter
   in the attribute.

3. Bidirectional `@differentiable(bidirectional, tangent: ..., adjoint: ...)`

   This option says that the function is both forward-mode differentiable and
   revese-mode differentiable. The compiler will expect both the tangent
   function and the adjoint function to be specified later in this attribute.

4. Constant `@differentiable(constant)`

   By definition, constant functions always have zero derivatives and are
   smooth. So differentiating this function will result into a vector (or
   vectors, when the function has multiple differentiation arguments) with
   the same dimensionality as each differentiation argument.

5. Linear `@differentiable(linear)`

   By definiton, a linear map is always a unary function and its Jacobian is
   the matrix associated with this linear transformation itself. In other
   words, both its differential and its pullback are itself.

#### Associated Functions

As explained, differentiabilities have different functional requirements.

1. `forward` differentiability

   When the differentiability is `forward`, the compiler expects a `tangent:`
   label in the attribute followed by the identifier (qualified or unqualified)
   of the tangent function that is to be associated with the original function.
   If the original function declaration has type `(T0, ..., Tn) -> U`, then
   the expected type of the tangent function is `((T0, T0), ..., (Tn, Tn)) ->
   (U, U)`. As we can see, every argument of the original function has become
   a "dual number" in the tangent function represented as a tuple. The first
   element of such a tuple is the original argument, the second argument the
   forward-propagated directional derivatives, namely the the "vector" in
   "Jacobian-vector product". The result of the tangent function is also a
   "dual number", a tuple of the original result and the directional
   derivatives. If any of the original arguments is marked as `@nondiff`, it
   will not become a dual number in the tangent function's argument list but
   will remain as the original argument itself.

2. `reverse` differentiability

   When the differentiability is `reverse`, the compiler expects an `adjoint:`
   label in the attribute followed by the identifier (qualified or unqualified)
   of the adjoint function that is to be associated with the original function.
   If the original function declaration has type `(T0, ..., Tn) -> U`, then
   the expected type of the adjoint function is `(T0, ..., Tn, U, U) -> (T0,
   ..., Tn)`. As we can see, the first `n` arguments to the adjoint function,
   `T0, ..., Tn`,  are the original arguments. The next argument is the
   original function's result. The last argument is the back-propagated
   partial derivatives at the original function's result,
   namely the "vector" in "vector-Jacobian product". The result of the
   adjoint function contains partial derivatives at each argument, if the
   argument has not been marked as `@nondiff`.

3. `bidirectional` differentiability

   When the differentiability is `bidirectional`, the compiler expects both
   `tangent:` and `adjoint:` labels in the attribute each followed by the
   identifier (qualified or unqualified) of the respective function
   that is to be associated with the original function.

4. Other differentiabilities

   Other differentiabilities such as `constant` and `linear` do not require
   any associated functions. However, the users can choose to specify
   tangent/adjoint function(s) for their own purposes such as custom
   optimizations.

#### Differentiation order

When a function is marked as `@differentiable`, Swift assumes it to be
[smooth](https://en.wikipedia.org/wiki/Smoothness), i.e. differentiable at all
orders, unless `once` is specified in the attribute, in which case Swift will
not guarantee any higher-order differentiability. If their associated functions
(tangent or adjoint) are serialized, then their derivatives _may_ be
differentiable via a separate code transformation.

Differentiabilities `linear` and `constant` guarantee smoothness, and
they do not have to be serialized whatsoever because their derivatives do not
depend on any code transformation.

`forward` and `reverse` transitively require the tangent function and the
adjoint function, respectively, to be smoooth with respect to the original
arguments. When compiling such declarations, Swift will verify the
tangent/adjoint function is also smooth by static analysis. If they are not
smooth, the compiler will error out, prompting the user to insert `once` in the
`@differentiable` attribute.

Example 1. Linear functions are differentiable at any order.

```swift
@differentiable(linear)
func linearFn(x: Vector<Float>) -> Float {
    ...
}
```

Example 2. A forward-mode primitive-differentiable function whose tangent is closed-form
and smooth is differentiable.

```swift
// Okay, the tangent function is smooth.
@differentiable(forward, tangent: tangentFoo)
func foo(_ x: Vector<Float>) -> Float {
    return Vector(repeating: sin(x), dimensionality: [2, 3])
}

func tangentFoo(_ dualX: (Float, Float), 
                originalValue: Vector<Float>) -> Vector<Float> {
    let (x, dx) = dualX
    // Smooth because `Vector.init(repeating:dimensionality:)`, `*`, `sin` and `cos` 
    // are all declared `@differentiable` and are smooth.
    return Vector(repeating: cos(x) * dx, dimensionality: [2, 3])
}
```

Example 3. A reverse-mode primitive-differentiable function whose tangent is not
smooth.

```swift
@differentiable(forward, adjoint: adjointBar)
func bar(_ x: Vector<Float>) -> Float {
    return sin(x)[0]
}

func adjointBar(_ x: Vector<Float>, y: Float, adjoint: Float) -> Vector<Float> {
    var ∂y∂x = Vector<Float>(repeating: 0, dimensionality: x.dimensionality)
    ∂y∂x[0] = cos(x[0]) * adjoint
    return ∂y∂x
}
```
```console
test.swift:3:35: error: adjoint function `adjointBar` does not support higher-order 
differentiation because it is not smooth; would you like to add `once` to declare the
function as non-smooth?
  @differentiable(reverse, adjoint: adjointBar)
                                    ^~~~~~~~~~
test.swift:8:6: note: `adjointBar` is defined here
  func adjointBar(_ x: Vector<Float>, y: Float, adjoint: Float) -> Vector<Float> {
       ^~~~~~~~~~
test.swift:10:9: note: operation is not differentiable
      ∂y∂x[0] = cos(x[0]) * adjoint
          ^~~~~~~~~~~~~~~~~~~~~~~~~
```

Part 3: Basic Differentiation
-----------------------------

### Most Important Cases: Gradient and Derivatives

```ebnf
derivatives-operator = '#derivatives'
gradient-operator = '#gradient'
raw-differential-operator = derivatives-operator | gradient-operator
autodiff-argument-index-specifier = '.' integer-literal
autodiff-expression =
    differential-operator '(' expression [ ',' 'wrt' ':' autodiff-argument-index-specifier ] ')'
expression = autodiff-expression
```

Example:
```swift
func f(_ x: Vector<Float>, _ w: Vector<Float>) -> Float {
   return x • w
}

#derivatives(f) // (T0, T1) -> (U) -> (U, (T0, T1))
#pullback(f) // (T0, T1) -> (U, (U) -> (T0, T1))
```

### Embrace Generality: Vector-Jacobian Products and Jacobian-Vector Products

```ebnf
jvp-operator = '#differential'
vjp-operator = '#pullback'
raw-differential-operator = jvp-operator | vjp-operator
```

Example:
```swift
// A random generic function that is differentiable.
func f<T0, T1, U>(_ x: T0, _ y: T1) -> U
    where T0: Differentiable, T1: Differentiable, U: Differentiable {
    return someDifferentiableFunction(20, x + y)
}

#differential(f) // (T0, T1) -> (U) -> (U, (T0, T1))
// Description:
//   (T0, T1)       ->  (U)    ->   (U,          (T0, T1))
//    ^~~~~~             ^           ^           ^~~~~~~~
//  original args      vector      result    Jacobian-vector products

#pullback(f) // (T0, T1) -> (U, (U) -> (T0, T1))
// Description:
//   (T0, T1)       ->  (U,     (U)      ->  (T0, T1))
//    ^~~~~~             ^       ^           ^~~~~~~~
//  original args     result   vector   vector-Jacobian products
```

### How It Works

The compiler type-checks a `#gradient(f)`, as well as other differential
operators, by searching for the closest match given the contextual type. `f` is
expected to have a definition to be differentiable, and thus cannot be an
closure whose body is opaque to the compiler. If so, Swift reports an error.

Later in the compilation pipeline, the compiler recursively transforms the code
of `f` to its gradient function `∇f` (or other functions in other modes of
differentiation), and replaces `#gradient(f)` with `∇f`. Everything composes
together naturally. Now, differentiation works.

### AD in Action

Automatic Differentiation based on raw differential opreators is already
available and being incubated temporarily on [the "tensorflow" branch of
Swift](https://github.com/apple/swift/tree/tensorflow). Swift for TensorFlow
[development
toolchains](https://github.com/tensorflow/swift/blob/master/Installation.md) and
[tutorials](https://github.com/tensorflow/swift-tutorials/blob/master/iris/swift_tensorflow_tutorial.ipynb)
are available for trying out this feature.

Part 4: Generalized Differentiability
-------------------------------------

Automatic differentiation relies on the definition (body) of a function to be
able to differentiate it. Differential operators like `#gradient` trigger the
differentiation of a function, and the differentiability of the function is
determined as differentiation goes. This works perfectly so far, but has a
number of problems.

### Issues with Definition-Based Differentiability

#### Syntactic Weirdness
   
Raw differential operators adopt the pound-keyword syntax, which has been
previously used for accessing compiler builtins, e.g. `#file` and `#dsohandle`,
referring to IDE-specific objects, e.g. `#colorLiteral` and `#imageLiteral`, and
interoperating with "stringly-typed" Objective-C key paths, e.g.
`#keyPath(...)`. The pound-keyword syntax does not have native parsing support
for syntactic features like trailing closures, so it is hard to make the closure
code short under differential operators like `#gradient`.

Example:
```swift
// Ideal
let dydx = gradient { x in
    sin(x) + cos(x)
}

// Reality
let dydx = #gradient({ x in
    sin(x) + cos(x)
})
```

#### A Higher-Order Function, But Not Quite

When we introduced AD in Swift earlier in this document, we defined the
differential operator as a higher-order function. Type checking and type
inference were just expected to work like any other functions.

However, since the compiler needs to reject functions that are not
differentiable and differentiability is not part of the type system, even we
were to redefine `#gradient` as a higher-order function named `gradient(of:)`,
the compiler will still have to maintain dedicated knowledge about this
function in order to reject invalid arguments.

#### Cross-Module Differentiability, Without Inlining

As of now, the differentiability of a function is determined solely through
two tests:
- Is the function a primitive-differentiable function (`@differentiable`)?
- Can the function's body be differentiated in the differentiation mode
  associated with the differential opreator applied?

This simple system works perfectly when differentiating concrete functions
defined in a local module, but does not allow differentiation of opaque function
values or methods required by protocols. While being free of serialization is
not a strict requirement for numerical computing libraries, not supporting
differentiation on protocol requirements fundamentally obstructs composable
high-level APIs that rely on AD, such as machine learning model APIs.

#### Opaque Closures are Non-Differentiable

There is no way to define a high-order function that differentiates its argument
using `#gradient`. Here's an example:

```swift
func foo(_ f: (Float) -> Float) -> Float {
    return #gradient(f)(0)
}
```

```console
test.swift:2:22: error: cannot differentiate an opaque closure
    return #gradient(f)(0)
           ~~~~~~~~~~^~
test.swift:1:12: note: value defined here
func foo(_ f: (Float) -> Float) -> Float {
           ^~~~~~~~~~~~~~~~~~~
```

Closure arguments and dynamic dispatch are non-differentiable through direct
source code transformation. The compiler does not statically know where `foo` is
coming from, nor can it delegate the task of differentiation of argument `f` to
each callsite of `foo` because it cannot be expressed in the type system.

### Solution: Differentiability in Function Types

We define a new formalization of differentiability in Swift's type system,
including an `@autodiff` function type attribute, an extension to functions'
layout, and new syntax for selecting differentiable arguments.

#### The `@autodiff` function type attribute

#### Conversion Between Differentiabilities

| Convertible to: | None | Linear | Constant | Forward | Reverse | Bidirectional |
|-----------------+------+--------+----------+---------+---------+---------------|
| None            | ✔    |        |          |         |         |               |
| Linear          | ✔    | ✔      |          | ✔       | ✔       | ✔             |
| Constant        | ✔    | ✔      | ✔        |         |         |               |
| Forward         | ✔    |        |          | ✔       |         |               |
| Reverse         | ✔    |        |          |         | ✔       |               |
| Bidirectional   | ✔    |        |          | ✔       | ✔       |               |

Part 5: True Differential Operators
-----------------------------------

### Derivatives and Gradient

```swift
/// Computes derivatives of `body` at scalar `x`.
func derivatives<T : FloatingPoint, R : Differentiable>(
    at x: T, in body: @autodiff(forward) (T) throws -> R
) rethrows -> R.TangentVector {
    let (y, dydx) = #differential(body)(x)(1) // seed = dx/dx = 1
    return dydx
}
```

```swift
/// Computes the gradient of `body` at `x`.
func gradient<T : Differentiable, R : FloatingPoint>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> T.CotangentVector {
    let (y, pullback) = #pullback(body)(x)
    return pullback(1) // seed = ∂y/∂y = 1
}
```

Example: Train a simple 2-layer perceptron. The snippet computes the gradient
w.r.t. each parameter at each training step, prints a loss, and optimizes
parameters.

```swift
struct Parameters : ParameterGroup {
    var w1 = Tensor<Float>(randomNormal: [784, 30])
    var b1 = Tensor<Float>(zeros: [30])
    var w2 = Tensor<Float>(randomNormal: [30, 10])
    var b2 = Tensor<Float>(zeros: [10])
}

var params = Parameters()
let minibatches = Dataset(...)
var optimizer = StochasticGradientDescent()
for (x, ŷ) in minibatches {
    let grads = gradient(at: params) { params in
        let h1 = tanh(matmul(x, params.w1) + params.b1)
        let y = sigmoid(matmul(h1, params.w2) + params.b2)
        let loss = (y - ŷ).squared().mean()
        print("Loss is \(loss)")
        return loss
    }
    optimizer.fit(&params, gradients: grads)
}
```


### Preserving Original Result

Since the trailing closure as an argument to `gradient(at:in:)`, the forward
computation is just as customizable as within operator-overloading AD systems.
Users can do whatever they want to intermediate values or the result in the
primal computation.

That said, we would like to provide a way to have the differentiation API return
the original result directly. Because of Generalized Differentiability, these
APIs can be defined entirely as library functions using primitive differential
operators.

```swift
/// Computes `body(x)` and derivatives of each scalar output of `body` at `x`.
func valueWithDerivatives<T: FloatingPoint, R: Differentiable>(
    at x: T, in body: @autodiff(forward) (T) throws -> R
) rethrows -> (value: R, derivatives: R.TangentVector) {
    return #differential(body)(x)(1)
}

/// Computes `body(x)` and the gradient of `body` at `x`.
func valueWithGradient<T: Differentiable, R: FloatingPoint>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (value: R, gradient: T.CotangentVector) {
    return #differential(body)(x)(1)
}
```

### Jacobian-Vector Products, Vector-Jacobian Products, and Jacobian

Jacobian-vector products (forward-mode) and vector-Jacobian products
(reverse-mode) are extremely useful differential operators for lots of tasks in
numerical computing.

```swift
/// Computes Jacobian-vector products of `body` at `x`.
func jacobianVectorProducts<T : Differentiable, R : Differentiable>(
    at x: T, vector: T,
    in body: @autodiff(forward) (T) throws -> R
) rethrows -> R {
    return #differential(body)(x)(vector)
}

/// Computes the vector-Jacobian products of `body` at `x`.
func vectorJacobianProducts<T : Differentiable, R : Differentiable>(
    at x: T, vector: R,
    in body: @autodiff(reverse) (T) throws -> R
) rethrows -> T {
    return #pullback(body)(x)(vector)
}
```

### Differentials and Pullbacks

In some cases, some computational tasks rely on fully extensible differential
operators as well as maximum efficiency, e.g. computing vector-Jacobian products
and also the original function's result. Luckily, the two operators we mentioned
in the very beginning when we introduced Jacobians are the ones we need:
differential and pullback.

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
    let (y, df) = pullback(at: x) { x in
        cos(sin(x))
    }

    df(1) // dy/dx
    df(#gradient(log)(t)) // dy/dt
    df(gradient(at: t, in: log)) // dy/dt
    ```

### Hessian-Vector Products

Second-order optimization methods in machine learning make use of
[Hessians](https://en.wikipedia.org/wiki/Hessian_matrix) and Hessian-vector
products, which can be hard to compute. Many AD libraries such as Autograd
already support Hessians by supporting arbitrarily nested
forward-mode/reverse-mode differentiation. Hessian-vector products can be
efficiently computed by applying "forward-on-reverse", namely applying the
composition of the forward-mode differential opreator and the reverse-mode
differential operator on a function.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\mathbf{H_f}(\mathbf{x})\mathbf{v}&space;=&space;\mathbf{J}_{\nabla&space;\mathbf{f}}(\mathbf{x})\mathbf{v}" title="\mathbf{H_f}(\mathbf{x})\mathbf{v} = \mathbf{J}_{\nabla \mathbf{f}}(\mathbf{x})\mathbf{v}" />
</p>

Just like other differential operators, we can define the Hessian-vector
products operator in a simple functional way.

```swift
func hvp<T, R>(_ f: @autodiff (T) -> R, vector: T) -> (T) -> T
    where T: Differentiable, R: FloatingPoint {
    return differential(gradient(f))
}
```

Nested differentiation without a careful implementation is prone to a bug known
as purturbation confusion
[[1]](http://www.bcl.hamilton.ie/~qobi/nesting/papers/ifl2005.pdf)
[[2]](https://arxiv.org/abs/1211.4892). Language-integrated AD in Swift will
enforce tagging in compiler-generated code to guarantee the correctness of
higher-order derivatives.

### Standard Library or an `AutomaticDifferentiation` Module?

Earlier in this document, we discussed enhancements to standard library
protocols and extensions to the standard library to model differentiable types.
These protocols are general enough for standard library types such as floating
point scalars (`Float`, `Double`, and `Float80`) and potentially [SIMD
vectors](https://github.com/apple/swift-evolution/blob/master/proposals/0229-simd.md).
However, in any general-purpose programming language, there is always a question
of how much math the standard library should have.

We do not the Swift standard library is a place for .
It can be easily defined by the user, and can be offered through a separate
`AutomaticDifferentiation` module in Swift so that it won't interfere with
other general-purpose functions.

Part 6: Generalized Types for Differentiation
---------------------------------------------

### Request for Future-Proof Design

There are three important use cases of such a generalization.

1. Customizable weight type

   Orthogonal weight matrixes have shown advantages in neural network training
   [[1]](https://arxiv.org/abs/1702.00071)
   [[2]](https://arxiv.org/abs/1709.06079). When differentiating through these
   networks, gradients with respect to weights will no long stay orthogonal -
   instead, they are skew-symmetric matrices. While we can represent both
   orthogonal matrices and skew-symmetric matrices as values of a `Matrix` or
   `Tensor` type and programmatically ensure its orthogonality, some researchers
   have been seeking a way to represent this natively in the type system of a
   programming language and still have AD produce the correct derivative.

2. Quantized training

   Quantization techniques store and calculate numbers in more compact formats,
   i.e. a fixed-point data type. Conceptually, a quantized tensor for a
   real-valued `Tensor` can be defined as the following struct:

   ```swift
   public struct Quantized<Dequantized: Quantizable, QuantizedScalar: FixedWidthInteger>
       var data: Quantizable
       var range: Range<QuantizedScalar>
       var scale: QuantizedScalar
       var zeroPoint: Int
   }
   ```

   We can think of a scenario where the developer defines a neural network as a
   function whose parameters are of type `QuantizedFloat`. When training
   parameters to this neural network, the function needs to be

3. Generic optimizers

   Optimization problems in machine learning is generalized by 

   This, in fact, has been exactly what people have been doing when writing machine
   learning optimizers, but the conversion has often been implicit because in most
   cases the tangent space is equal to the cotangent space in a vector space
   scenario. With this generalization, people will be able to write general
   optimizers that is both practically general and mathematically correct.

### The New `Differentiable` Protocol

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

When a differentiable manifold is a vector space, it's tangent space is usually 
itself. In these cases, we simply define `moved(toward:)` as vector addition. 

```swift 
public extension Differentiable 
    where Self : VectorNumeric, TangentVector == Self { 
    func moved(toward direction: TangentVector) -> Self { 
        return self + direction 
    } 
} 
``` 

### Deriving Conformances to `VectorNumeric` and `Differentiable`

It is very common for numerical computing to deal with lots of parameters, each
of which is a vector or a matrix. In these cases, instead of manually specifying
each input in a differential opreator's parameter list, users would often like
to differentiate through structures and obtain a stucture of partial
derivatives. It is important for the Swift to provide derived conformances for
core protocols for numerical computing: `Differentiable` and `VectorNumeric`.

Mathematically, it is straightforward to represent product types. A struct or
tuple in Swift corresponds to a product of sets; an enum in Swift
corresponds to an addition of sets.

```swift
struct Parameters : VectorNumeric, Differentiable {
    var a: Vector<Float>
    var b: Float
}
```

Struct `Parameters` is equivalent to a product of sets `Vector<Float>` and
`Float`, or a product of a real vector space `ℝⁿ` and a scalar field `ℝ`, namely
`ℝⁿ ⨯ ℝ`, which is also a vector space. To make `Parameters` obtain the traits
of a vector space, we extend the compiler to derive a conformance to
`VectorNumeric` similar to how `Codable` and `Hashable` conformances are
derived. When a conformance clause is given in the current file and when all
stored properties conform to `VectorNumeric` with the same `ScalarElement`, the
compiler synthesizes AST to make this type conform, with all protocol requirements
applying property-wise.

After deriving conformances to `VectorNumeric`:

```swift
struct Parameters : VectorNumeric {
    var a: Vector<Float>
    var b: Float

    // derived:
    typealias ScalarElement = Float

    // derived:
    struct Dimensionality {
        var a: Vector<Float>.Dimensionality
        var b: Float.Dimensionality
    }

    // derived:
    func + (lhs: Parameters, rhs: Parameters) -> Parameters {
        return Parameters(a: lhs.a + rhs.a, b: lhs.b + rhs.b)
    }
    // ...
}
```

In order for `Parameters` to be differentiable, it must also need to conform to
`Differentiable`. Deriving conformances to `Differentiable` can follow the same
rules.

```swift
struct MyShapes : Differentiable {
    var a: Circle // conforms to Differentiable
    var b: Square // conforms to Differentiable
}
```

After deriving conformances to `Differentiable`:

```swift
struct MyShapes : Differentiable {
    var a: Circle
    var b: Square

    // derived:
    struct TangentVector : VectorNumeric {
        var a: Circle.TangentVector
        var b: Square.TangentVector
    }
    // derived:
    struct CotangentVector : VectorNumeric {
        var a: Circle.CotangentVector
        var b: Square.CotangentVector
    }

    // derived:
    func moved(toward direction: TangentVector) -> MyShapes {
        return MyShapes(a: a.moved(toward: direction.a),
                        b: a.moved(toward: direction.b))
    }

    // derived:
    func tangentVector(from cotangent: CotangentVector) -> TangentVector {
        return TangentVector(a: a.tangentVector(from: cotangent.a)
                             b: b.tangentVector(from: cotangent.b))
    }
}
```

With derived conformances to these protocols, the user can now write arbitrarily
nested structs of differentiable manifolds, and make them differentiable with
trivial effort, greatly simplifying the development.

### Generalized Differential Operators

In the new `Differentiable` protocol, we added `Tangent` and `Cotangent` types
 to represent the type of Jacobian-vector products and vector-Jacobian products,
 respectively. We make the following changes to the existing differential
 operators we introduced.
 - Differnetial opreators that return `T` as a forward-differentiated derivative
   will return `T.Tangent` instead.
 - Differential operators that return `T` as a reverse-differentiated derivative
   will return `T.Cotangent` instead.
 - Vectors `T` for computing Jacobian-vector products will become `T.Tangent`.
 - Vectors `T` for computing vector-Jacobian products will become `T.Cotangent`.

 Here we list a few updated differential operators.

#### Jacobian-Vector Products and Vector-Jacobian Products

Jacobian-vector products (forward-mode) and vector-Jacobian products
(reverse-mode) are extremely useful differential operators for lots of tasks in
numerical computing.

```swift
/// Computes Jacobian-vector products of `body` at `x`.
func jacobianVectorProducts<T : Differentiable, R : Differentiable>(
    at x: T, vector: T.TangentVector,
    in body: @autodiff(forward) (T) throws -> R
) rethrows -> R.TangentVector {
    return #differential(body)(x)(vector)
}

/// Computes the vector-Jacobian products of `body` at `x`.
func vectorJacobianProducts<T : Differentiable, R : Differentiable>(
    at x: T, vector: R.CotangentVector,
    in body: @autodiff(reverse) (T) throws -> R
) rethrows -> T.CotangentVector {
    return #pullback(body)(x)(vector)
}
```

### Differential and Pullback

```swift
/// Computes the differential of `body` at `x`.
func differential<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (T.TangentVector) -> (originalResult: T, derivative: R.TangentVector) {
    return #differential(body)(x)
}

/// Computes the original value of `body(x)` and the pullback (chainable gradient)
/// at `x`.
func pullback<T : Differentiable, R : Differentiable>(
    at x: T, in body: @autodiff(reverse) (T) throws -> R
) rethrows -> (originalResult: T, pullback: (R.CotangentVector) -> T.CotangentVector) {
    return #pullback(body)(x)
}

### Solution

1. Custom weight type

2. Quantized Training

   ```swift
   // `Quantized` is a vector space when the dequantized type is one.
   extension Quantized: VectorNumeric where Dequantized: VectorNumeric {
       typealias ScalarElement = Dequantized.ScalarElement
       static func + (lhs: Quantized, rhs: Quantized) -> Quantized {
           // Custom code: Dequantize, add, and requantize!
       }
       static func * (lhs: ScalarElement, rhs: Quantized) -> Quantized {
           // Custom code: Dequantize, add, and requantize!
       }
   }

   // `Quantized` is a differentiable manifold when the dequantized type is one.
   extension Quantized: Differentiable where Dequantized: Differentiable {
       typealias TangentVector = Dequantized.TangentVector
       typealias CotangentVector = Dequantized.CotangentVector

       func moved(toward tangent: Dequantized.TangentVector) -> QuantizedTensor {
           // Custom code: Dequantize, optimize, and requantize!
       }
   }
   ```

   One would expect to differentiate a function of type
   `(Quantized<Tensor<Float>, Int8>) -> U` to train parameters. However, we
   never want gradients at each parameter to be a quantized number - instead,
   the range and zero point are often recomputed for each parameter update.

Part 7. Customizable differentiation
------------------------------------

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
    @differentiable(forward, wrt: self, tangent: adjointWithCustomizedDerivatives)
    func withCustomizedDerivatives(
        using body: @nondiff (TangentVector?) -> TangentVector?
    ) -> Self {
        return value
    }

    internal func adjointWithCustomizedDerivatives(
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
    prediction = prediction.withCustomizedDerivatives { grad in
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
that wraps `withCustomizedDerivatives(using:)` and returns a `Bool`, so that
later code can decide whether to `break` from the loop based on the return value
from that API.

Applications
------------

Future Directions
-----------------

Conclusions
-----------

Acknowledgements
----------------

The author would like to thank Dan Zheng, Chris Lattner, Alex Wiltschko, Bart
van Merriënboer, Gordon Plotkin, Dougal Maclaurin, Matthew Johnson, Casey Chu,
and Tim Harley for their input to the design of this powerful language feature.
