---
tags:
  - Banach-space-theory
parent: "[[Bounded Inverse Theorem]]"
children:
---


# Statement
---
- **Theorem (<u style='color:red'>Closed Graph Theorem</u>)**:  
---
<u>Pre-requisites</u>:
- Let $\mathcal{X, Y}$ be Banach Spaces.
- $T:\mathcal{X}\to\mathcal{Y}$ is a _linear operator_.

<u>Then</u>: 
$$
T \text{ continuous (bounded)} \iff T \text{ closed}
$$

# Proof
---
### 1. $(i)$ Continuity implies closedness
---
Suppose that we have $\{x_n\}\subset\mathcal{X}$ such that $x_n\to x\in\mathcal{X}$. Then, we have:
$$
\begin{align*}
\|Tx_n - Tx\|_\mathcal{Y} &= \|T(x_n - x)\|_\mathcal{Y} \\
&\le \|T\|_{op} \cdot \|x_n - x\|_\mathcal{X}
\end{align*}
$$
Since $\|x_n-x\|_\mathcal{X}\to0$ we have $\|Tx_n - Tx\|_\mathcal{Y}\to0$. Hence $Tx_n\to Tx$ and the linear operator $T$ is indeed closed.

### 2. $(ii)$ Closedness implies continuity (boundedness)
---
We define the graph of operator $T$ as followed:
$$
\mathcal{X}\times\mathcal{Y} \supset \Gamma(T) = \Big\{
(x, Tx) : x \in \mathcal{X}
\Big\}
$$

$T$ is closed implies that $\Gamma(T)$ is a closed subspace of $\mathcal{X}\times\mathcal{Y}$, which is a Banach Space. Hence $\Gamma(T)$  itself is a Banach Space with respect to the norm:
$$
\|(x, Tx)\|_{\Gamma(T)} = \|x\|_\mathcal{X} + \|Tx\|_\mathcal{Y}
$$

Define $P:\Gamma(T)\to \mathcal{X}$ such that $P(x, Tx)=x$. We can easily verify that $P$ is _injective and linear_. We can also verify that it is _bounded_ (with constant $C=1$):
$$
\|P(x, Tx)\|_{\mathcal X} = \|x\|_\mathcal{X} \le \|x\|_\mathcal{X} + \|Tx\|_\mathcal{X} = \|(x, Tx)\|_{\Gamma(T)}
$$

Hence, by [[Bounded Inverse Theorem]], we have $P^{-1}$ is bounded. Which means, there exists $\tilde C > 0$ such that $\|P^{-1}(x)\|_{\Gamma(T)}\le \tilde C\|x\|_\mathcal{X}$. We have:
$$
\begin{align*}
\|P^{-1}(x)\|_{\Gamma(T)} &= \|(x, Tx)\|_{\Gamma(T)} \\
&= \|x\|_\mathcal{X} + \|Tx\|_\mathcal{Y} \\
&\le \tilde C \|x\|_\mathcal{X} \\
\implies \|Tx\|_\mathcal{Y} &\le (\tilde C - 1)\cdot \|x\|_\mathcal{X}
\end{align*}
$$

Hence, $T$ is bounded (with constant $\tilde C - 1$).
