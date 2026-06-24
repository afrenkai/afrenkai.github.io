# On Second Order Optimizers

Recently, there was a little bit of debate in one of my circles about using [Muon](https://kellerjordan.github.io/posts/muon/) or other second order optimizers for [LoRA](https://arxiv.org/abs/2106.09685) adapters of LMs (or other tasks). However, in my mind, I can't get past the fact that LoRAs are supposed to be translation invariant. Existing papers do not do a fair job or "apples to apples" comparisons of AdamW to Muon, where LRs are purposely nerfed and speed is tossed aside for accuracy (Riemannion)[https://arxiv.org/abs/2507.12142], or have other issues. First, it is important to build a background, as I have learned a lot while writing and researching this. 

## 
LoRA has what is called a "gauge symmetry": [sparse wikipedia article here](https://en.wikipedia.org/wiki/Gauge_symmetry_(mathematics)). For any invertible $Q \in GL(r)$: vector bundles $A, B$ and $AQ, Q^{-11}B$ produce the same weights $W=AB$. This can be interpreted as 
A: the landscape of the loss being degenerate along the direction of the gauge
and B: the hessian being completely singluar in the space of $A, B$ by its very construction. The effect of this symmetry is actually very interesting in itself. The map $\phi: (A, B) \rightarrow W = AB$ has a Jacobian $\mathcal{J}$
This jacobian can be written as:

$$
\mathcal{J} = [(B^T \otimes I_m), (I_n \otimes A)] \in \mathbb{R}^{mn \times r(m+n)}
$$

What we can find from $J$ is that its null space has all of the perturbations of $(QA, -QB)$ for all $Q \in \mathbb{R}^{r \times r}. With dimension $r^2$, the hessian is rank deficient by at least $r^2$, (in A,B space), **no matter the loss! **

$$\nabla^2 = \mathcal{J}^T H_W \mathcal{J} + \Sigma_G$$, where $\Sigma_G$ is a first order correction involving the gradient $G = \frac{\partial{L}}{\partial{W}}$.

The non-degeneration portion of the subspace is of dimension $r+mn - r^2 = r(m+n -r)$. So, all of the comparisons an optimizer can actually theoretically do are in this dimension. 

## Cross-Block hessians

The actual hessian, not regarding the directions of the gauge symmetries, can be constructed in a "block":
$$H_{\theta} = 
\begin{bmatrix}
H_{AA} & H_{AB}\\
H_{BA} & H_{BB}\\
\end{bmatrix}$$
The diags are relatively easy to compute:
$$
H_AA \simeq (B \otimes I_m)^T H_W (B \otimes I_m)$$
$$H_BB \simeq (I_n \otimes A)^T H_W (I_n \otimes A)$$
However, the off-diag cross-block is where the actual interesting revelation is. In order to approximate it, you need to take $\frac{dG_A = DB^T}{dB}$. With this term, you can approximate $H_AB$
$$H_AB \simeq (I_n \otimes A^T) H_W (B \otimes I_m) + (G^T \otimes I_r)$$. 
The last term, or the result of the first order differentiation is **directly proprtional to the magnitude of the gradient**. This means that when fine-tuning, it is large, and when converging it is small. 
The conclusion one can draw from this is that any optimizer that treats A and B independently, or precontions on the block-diagonals completely misses the importance of the off-diagonal of the hessian. 
A fair question would be: "does that term even matter?". A lot of people explicitly remember a concerted effort to remove hessians from pretraining, and generally gradients do not get large enough to actually matter. However, mathematically it comes up when $$\frac{\|H_{AB}}\|}{min(\|H_{AA}\|, \H_{BB}\|)} is large. This mainly occurs (in my very limited experience) a large gradient, which happens early in training, a hard transfer learning task, or an ill-conditioned problem (in which case, the optimizer is the least of your problems). 

At $B=0$, $H_{AB} = (G^T \otimes I_r)$ which will be big (you have probably 0 init your weights, or are similarly far from your target distribution). But, as $B$ gets larger, the Kronecker term domniates and $G \rightarrow 0$. Of course, this is never negligible, but optimizers that precondition on the off-diagonal make an error. 
