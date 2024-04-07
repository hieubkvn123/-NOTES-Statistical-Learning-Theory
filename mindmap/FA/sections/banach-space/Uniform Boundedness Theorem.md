---
tags:
  - Banach-space-theory
parent: "[[Baire's theorem]]"
children:
---
---
---


# Statement
---
- **Theorem (<u style='color:red'>Uniform Boundedness Theorem/Banach Steinhaus Theorem</u>)**:  
---
<u>Pre-requisites</u>:
- Let $\mathcal{X}$ be a _Banach space_, $Y$ be a normed space.
- Let $\mathbb{B}(\mathcal{X}, Y)$ be the space of all _bounded linear operators_ from $\mathcal{X}\to Y$.
- Suppose that: for all $x\in\mathcal{X}$, $\|Tx\|_\mathcal{X}$ is finite for all $T\in\mathbb{B}(\mathcal{X}, Y)$.  
$$
\forall x\in \mathcal{X}, \ \exists C_x>0 : \sup_{T\in\mathbb{B}(\mathcal{X}, Y)}\|Tx\|_Y \le C_x 
$$

<u>Then</u>: The operator norm of all $T\in\mathbb{B}(\mathcal{X}, Y)$ is finite:
$$
\exists C>0 : \sup_{T\in\mathbb{B}(\mathcal{X}, Y)} \|T\|_{op} \le C
$$


# Proof
---
First of all, recall the definition of operator norm:
$$
T\in\mathbb{B}(\mathcal{X}, Y) : \|T\|_{op} = \sup_{\|x\|_\mathcal{X}\le 1} \|Tx\|_Y
$$

Define the following subsets of $\mathcal{X}$:
$$
X_n = \Bigg\{
x\in\mathcal{X} : \sup_{T\in\mathbb{B}(\mathcal{X}, Y)}\|Tx\|_Y \le n
\Bigg\}
$$

By assumption we have $\mathcal{X} = \bigcup_{n\in\mathbb{N}} X_n$.


#### *Claim* : For all $n\in\mathbb{N}$, $X_n$ is closed.
---
First, we realized that:
$$
X_n = \bigcap_{T\in\mathbb{B}(\mathcal{X}, Y)} \Big\{x \in \mathcal{X} : \|Tx\|_Y \le n\Big\}
$$
For all $T\in \mathbb{B}(\mathcal{X}, Y)$, we can prove that the set $X_n(T) = \Big\{x\in\mathcal{X}:\|Tx\|_Y\le n\Big\}$ is closed. Hence $X_n$, which is a union of closed set will indeed be closed. Since $T$ is a continuous linear map from a Banach Space, we have:
$$
\forall \{x_k\} \subset \mathcal{X}, x_k \to x \implies Tx_k \to Tx
$$
Hence, we have:
$$
\|Tx\|_Y = \lim_{k\to\infty} \|Tx_n\|_Y \le n \implies x\in X_n(T)
$$
---

Now that we have proved that $X_n$ is closed for all $n\in\mathbb{N}$. By [[Baire's theorem]], there exists $m\in\mathbb{N}$ such that $X_m$ contains an open ball. Suppose we have $\mathcal{B}(x, \epsilon)\subset X_m, \ \epsilon>0, x\in X_m$. We have:
$$
\begin{align*}
\|
\end{align*}
$$


# Intuition
---

|                                        | **$\color{cyan}x_1$** | **$\color{cyan}x_2$** | **$\dots$** | $\color{cyan}x_N$ |     | *$\color{yellow}\tilde x_1$* | *$\color{yellow}\tilde x_2$* | ***$\dots$*** | *$\color{yellow}\tilde x_{\tilde N}$* |
| -------------------------------------- | --------------------- | --------------------- | ----------- | ----------------- | --- | ---------------------------- | ---------------------------- | ------------- | ------------------------------------- |
| $\color{cyan}T_1$                      | $T_1x_1$              | $T_1x_2$              | $\dots$     | $T_1x_N$          |     | *$T_1\tilde x_1$*            | *$T_1\tilde x_2$*            | *$\dots$*     | $T_1\tilde x_{\tilde N}$              |
| $\color{cyan}T_2$                      | $T_2x_1$              | $T_2x_2$              | $\dots$     | $T_2x_N$          |     | *$T_2\tilde x_1$*            | *$T_2\tilde x_2$*            | *$\dots$*     | $T_2\tilde x_{\tilde N}$              |
| $\vdots$                               | $\vdots$              | $\vdots$              | $\ddots$    | $\vdots$          |     | *$\vdots$*                   | *$\vdots$*                   | *$\ddots$*    | $\vdots$                              |
| $\color{cyan}T_M$                      | $T_Mx_1$              | $T_Mx_2$              | $\dots$     | $T_Mx_N$          |     | *$T_M\tilde x_1$*            | *$T_M\tilde x_2$*            | *$\dots$*     | $T_M\tilde x_{\tilde N}$              |
| $\color{cyan}\text{Point-wise bound:}$ | $\le C_1$             | $\le C_2$             | $\dots$     | $\le C_N$         |     | $\le \tilde C_1$             | $\le \tilde C_2$             | $\dots$       | $\le \tilde C_{\tilde N}$             |

- Since we know that $\sup_{T\in\mathbb{B}(\mathcal{X}, Y)}\|T_x\|_Y$ is finite for all $x\in\mathcal{X}$, it is finite for the following subset of $\mathcal{X}$:
$$
\tilde{\mathcal{X}} = \Big\{ x \in \mathcal{X} : \|x\|_\mathcal{X} = 1 \Big\}
$$
- For all $\tilde x\in\tilde{\mathcal{X}}$, we have $C(\tilde x)>0$ such that $\sup_{T\in\mathbb{B}(\mathcal{X}, Y)}\|T\tilde x\|_Y\le C(\tilde x)$. Let:
$$
C = \sup_{\tilde x_* \in \tilde{\mathcal{X}}} C(\tilde x_*)
$$
- We have:
$$
\begin{align*}
\sup_{T\in\mathbb{B}(\mathcal{X}, Y)} \|T\|_{op} 
&=\sup_{T\in\mathbb{B}(\mathcal{X}, Y)} \Bigg\{ \sup_{\tilde x_* \in\tilde{\mathcal{X}}} \|T\tilde x_*\|_Y\Bigg\} \\
&= \sup_{\tilde x_* \in\tilde{\mathcal{X}}} \Bigg\{ \sup_{T\in\mathbb{B}(\mathcal{X}, Y)} \|T\tilde x_*\|_Y \Bigg\}  \\ \\
&\le \sup_{\tilde x_* \in \tilde{\mathcal{X}}} C(\tilde x_*) = C
\end{align*}
$$

- But heck, we still need to prove it using the axioms we have so far only which is [[Baire's theorem]].