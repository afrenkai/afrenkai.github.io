# "4D" Attention and Masked (Diffusion) Language Models

I am sure you know of attention. If not, I would encourage a do a little bit of reading and come back, since a lot of this hinges on prior knowledge. For this, I used the pytorch and Flash Attention Implementations as reference, as well as transformers. Mileage may vary with other frameworks. 

## Scaled Dot-Product Attention

Normal Scaled Dot Product Attention that we all know and love follows the following formula. 
\begin{equation}

\text{Given query length} L, \text{and key/value length} S, \text{tensors} Q \in \mathbb{R}^{L\times d_{k}}, K \in \mathbb{R}^{S \times d_{k}}, \text{and} V \in ℝ^{s \times d_{v}} \text{(where} d_n \text{) represents the feature dimension of the head)} are constructed. 
\end{equation}

write some bs here about sdpa



- there is a mask
[-inf, -inf, x
 -inf, something, something something, something, something
]


(b x n_heads x len(q) x len(k))


## Laziness of 2d Masks

normal impl uses mask of shape (batch, seq_len) which expands to (batch, n_heads_or_1, len_q, kv_len). 

assumes causal, padding-only

if bidirectional in a block, cross-attn to positions in a seq, block-diag w/ packing, etc.. shits itself

real 4d M is O((BHL^2)) and bound by the SDPA formula

solution is to not store the bias term at all and "describe" as a program. FlexAttn has mask_mod/BlockMask which can compile to block-sparse kernels and be used in flash-attn 4.



