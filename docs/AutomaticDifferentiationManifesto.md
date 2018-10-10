# The Automatic Differentiation Manifesto

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

The Swift for TensorFlow project aims to provide best-in-class support for AD -
including the best optimizations, best error messages in failure cases, and the
most flexibility and expressivity. To achieve this, we built support for AD
right into the Swift compiler. Additionally, since AD is important to the
broader scientific and numerical computing communities, we decided to build AD
as a generic feature that is completely orthogonal to the TensorFlow support -
the TensorFlow Swift library computes gradients using the AD features of the
Swift language itself.

## Why Swift needs Automatic Differentiation?



## Syntax

```
differentiable-attribute ::= '@differentiable' '(' differentiability ')'
differentiability ::= differentiation-mode | 'linear' | 'constant'
differentiation-mode ::= 
differentiation-mode ::= 
```

## Semantics


##


##

