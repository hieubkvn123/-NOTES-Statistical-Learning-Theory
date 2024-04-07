---
tags:
  - Banach-space-theory
parent: "[[Baire's theorem]]"
children: "[[Bounded Inverse Theorem]]"
---


# Statement
---
- **Theorem (<u style='color:red'>Open Mapping Theorem</u>)**:  
---
<u>Pre-requisites</u>:
- Let $\mathcal{X}, \mathcal{Y}$ be Banach Spaces.
- $T:\mathcal{X}\to\mathcal{Y}$ is a _surjective (onto) bounded linear_ operator.

<u>Then</u>: $T$ is an open map. Meaning:
$$
U\subset \mathcal{X} \text{ is open} \implies T(U) \subset \mathcal{Y} \text{ is open}
$$

# Proof
---
Let $U\subset\mathcal{X}$ be an open set. Then, there exists $x\in U, \epsilon > 0$ and we also denote $Tx=y$ (since $T$ is surjective, we can always find $y\in\mathcal{Y} : Tx=y$) such that:
$$
\begin{align*}
U &\supset \mathcal{B_X}(x, \epsilon) \\
\implies T(U) &\supset T(\mathcal{B_X}(x, \epsilon)) \\
&= T(x +  \epsilon\mathcal{B_X}(0,1)) \\
&= Tx + \epsilon T(\mathcal{B_X}(0,1)) \\
&= y + \epsilon T(\mathcal{B_X}(0,1))
\end{align*}
$$

By the [[Open Unit Ball Lemma]], there exists $\delta>0$ such that:
$$
T(\mathcal{B_X}(0,1)) \supset \mathcal{B_Y}(0,\delta)
$$

Hence, we have:
$$
\begin{align*}
T(U) \supset y + \epsilon T(\mathcal{B_X}(0,1))\supset y + \epsilon\mathcal{B_Y}(0,\delta) = \mathcal{B}(y, \epsilon\delta)
\end{align*}
$$

Since $T$ is surjective, we can always pick any $y\in T(U)$ and find $x\in U$ such that $Tx=y$ and construct the open ball about $y$ as done above.

Hence, for every $y\in T(U)$, there is an open ball $\mathcal{B_Y}(y, \kappa)\subset T(U)$ where $\kappa=\epsilon\delta$ as constructed above. 
