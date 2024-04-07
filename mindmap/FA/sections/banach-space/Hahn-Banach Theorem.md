---
tags:
  - Banach-space-theory
parent: "[[Zorn's Lemma]]"
children: 
comment: This is probably the part of functional analysis where I rarely touch. I just know Hahn-Banach theorem is there but I rarely use it in research. So I will not put any proof.
---


# Statement
---
- **Theorem (<u style='color:red'>Hahn-Banach Theorem</u>)**:  
---
<u>Pre-requisites</u>:
- Let $X$ be a real/complex norm space.
- Let $M\subseteq X$ be a linear subspace.
- Let $T:M\to \mathbb{F}$ be a linear bounded operator ($\mathbb{F}=\{\mathbb{R,C}\}$) on $M$. Meaning:
$$
\forall x\in M : |Tx| \le C\cdot \|x\|_M
$$

<u>Then</u>: There exists a continuous extension $T^*:X\to\mathbb{F}$ such that:
- $(i) \ \ T^*_{|M}:=T$.
- $(ii) \ |T^*x| \le C\cdot \|x\|_X, \ \ \forall x\in X$.

_Meaning the continuous extension $T^*$ is bounded by the same constant as $T$ on $M$_.


# Proof
---
- _There will be no proof for this theorem_.