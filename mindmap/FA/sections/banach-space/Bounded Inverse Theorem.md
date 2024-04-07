---
tags:
  - Banach-space-theory
parent: "[[Open Mapping Theorem]]"
children:
---


# Statement
---
- **Theorem (<u style='color:red'>Theorem name</u>)**:  
---
<u>Pre-requisites</u>:
- Let $\mathcal{X}, \mathcal{Y}$ be Banach Spaces.
- $T:\mathcal{X}\to\mathcal{Y}$ is an _injective (one-to-one + onto) bounded linear operator_.

<u>Then</u>: $T^{-1}$ exists and $T^{-1}\in\mathbb{B}(\mathcal{X}, \mathcal{Y})$.

# Proof
---
- We know $T^{-1}$ exists because $T$ is bijective. Further more, $T^{-1}$ is surjective so that we can apply the Open Mapping Theorem.
- By [[Open Mapping Theorem]], $T$ is an open map. Hence, we have:
$$
\forall V \subset \mathcal{Y} \text{ open} \implies T^{-1}(V) \subset \mathcal{X} \text{ open}
$$

- Hence, $T^{-1}$ is also an open map. Therefore, $T^{-1}\in\mathbb{B}(\mathcal{Y, X})$.