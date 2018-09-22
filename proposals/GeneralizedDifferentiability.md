# Automatic Differentiation: Generalized Differentiability

* Author: [@rxwei](https://github.com/rxwei)

## Abstract

Automatic differentiation relies on the source code of a function to be able to
differentiate it. Differential operators like `#gradient` trigger the
differentiation of a function, and the differentiability of the function is
determined as differentiation goes. This simple system works perfectly when
differentiating concrete functions defined in a local module, but does not allow
differentiation of opaque function values or methods required by protocols.
While the former is not a struct requirement in machine learning systems, the
latter fundamentally obstructs composable, protocol-oriented machine learning
APIs. This proposal introduces a new function type attribute in the Swift
language, `@differentiable`, which encodes a function's differentiability
throughout the language.

## Introduction

[Automatic differentiation in
Swift](https://github.com/tensorflow/swift/blob/master/docs/AutomaticDifferentiation.md)
is a compiler transform, and the differentiability of a function can be
statically determined based upon the following factors:

- the function's argument types and result type
- the data flow and control flow in the function body

Like in mathematics, differential operators are conceptually higher-order
funtions - they take a function and return a function that computes
vector-Jacobian products. In Swift, I built differential operators into the
programming language as a keyword. There are four differential operators in
today's system. To demonstrate their functionality, here's a simple real
function.

```swift
func foo<T, U>(_ x: T1, _ y: T2) -> U 
    where T1 : FloatingPoint, T2 : FloatingPoint, U : FloatingPoint
```

- `#gradient` produces a function that returns the vector-Jacobian products with
  evaluated at the specified arguments.
  ```swift
  #gradient(foo) // (T1, T2) -> (T1, T2)
  //                             ^    ^
  //                            d/dx d/dy
  ```

- `#valueAndGradient` produces a function that returns a tuple of the original
  function's result and the vector-Jacobian products evaluated at the specified arguments.
  ```swift
  #valueAndGradient(...) // (T1, T2) -> (U, (T1, T2))
  //                                     ^   ^    ^
  //                                f(x, y) d/dx d/dy
  ```

- `#chainableGradient` produces a function that takes a backpropagated
  vector-Jacobian product as an extra argument, and returns the vector-Jacobian
  products evaluated at the specified arguments.
  ```swift
  #chainableGradient(...) // (T1, T2, U) -> (T1, T2)
  //                           ^   ^  ^      ^    ^
  //                           x   y d?/dy  d?/dx d?/dy
  ```

- `#delayedGradient` produces a function that returns a tuple of the original
  function's resunt and a closure which takes a back-propagated vector-Jacobian
  product and returns the vector-Jacobian products evaluated at the specified arguments.
  ```swift
  #delayedGradient(...) // (T1, T2) -> (U, (U) -> (T1, T2))
  //                                    ^   ^      ^    ^
  //                                f(x, y) d?/dx d/dx d/dy
  ```
  
Differential operators above generalize most use case in the context of machine
learning.


## Differential operators

## Differentiability in protocols

```swift
/// A type whose values have parameters.
///
/// Instances of `Parameterized` types have parameters, represented as stored
/// properties. Stored properties are marked as parameters with the `@TFParameter`
/// attribute. The Swift compiler automatically generates a member struct type
/// `Parameters`, which includes all of the marked properties.
///
public protocol Parameterized {
  associatedtype Parameters : ParameterAggregate
  var parameters: Parameters { get set }
}

/// A neural network module.
///
/// Types that conform to `Module` represent functions that map inputs to
/// outputs. They may have an internal state represented by parameters, such as
/// weight tensors.
///
/// `Module` instances define an `prediction(for:using:)` method for mapping inputs to outputs.
public protocol Module : Parameterized {
  associatedtype Input
  associatedtype Output
  func prediction(for input: Input, using parameters: Parameters) -> Output
}

/// A differentiable neural network module.
///
/// Types that conform to `DifferentiableModule` represent differentiable
/// functions that map inputs to outputs.
///
/// `DifferentiableModule` instances define a `gradient` method that computes
/// the gradient with respect to an input and the instance's parameters.
public protocol DifferentiableModule {
  func gradient(for input: Input, 
                backpropagating seed: Output) -> (Input, Parameters)
}
```

## A new function representation
