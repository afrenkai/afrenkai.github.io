# VQVAE

- Vector Quantization: Map from $\mathbb{R}^m$ to $\mathcal{S} \in \mathbb{R}$ where $| \mathcal{S} | = K$ (assuming you want some $K$ classes) for vector output
- Alternatively, a map from $\mathbb{R}^m$ to $\left \{ 1 , \dots , K \right \}$ which outputs an integer or index.
- Based on this, we can create an “autoencoder” that trains on a dataset $X$
    - $Q$ maps $X$ to a latent representation $z$ like in a regular autoencoder but $z \in \left \{ 1 \dots K \right \}$ corresponding to a centroid $\mu_z$that data would belong to.
    - $P$ would map $z$ to some vector $\mu_z$ as the estimate for $x$
    - You would use SSD as the recon loss metric and use K-Means to optimize.
    - Can also make a discrete “VAE” where $Q$ maps $x$ deterministically to the closes centroid (in L2 norm)
        
        $$
        Q(z | x) \begin{cases} 1 \ \text{if} \  z = \argmin_{k} ||x - \mu_k||^2_2
        \\
        0 \ \text{else}
        \end{cases}
        $$
        
    - and $P$ maps $z$ to a gaussian centered at $\mu_z$:
        
        $$
        P(x|z) = \mathcal{N}(\mu_z, I)
        $$
        
    - It becomes trivial to train this with a Gaussian Mixture Model and MLE. You can enforce that $P(z) \sim Unif(1 \dots K)$ or each of the centroids is equally likely (uniformly distributed).
    - Once it is trained, you can very easily sample
        
        $$
        z \sim P(z) = Unif(1 \dots K)\\
        \hat{x} \sim P(x | z) = \mathcal{N}(\mu_z, I)
        $$
        
    - This “model” is actually pretty bad. This just adds noise to centroids in the higher dimensional space. But this idea does extend to the actual VQ-VAE
    - Given $X \in \mathbb{R}^{m}$:
        - you want to transform each little $x \in X$ into feature vector $h \in \mathbb{R}^{l \times d}$
            
            ![image.png](VQVAE/image.png)
            
        - then quantize $h$ into $h’$ using the nearest centroid(s) $e_z$
            
            ![image.png](VQVAE/image%201.png)
            
        - then transform $h'$ into $P(x | h') = P(x | z)$
        - Issue: This is basically still K-means. x is represented by 1 point $h'$ in latent space. If each x has just 1 point, then its recon must be the same for all other $x'$ that map to $h'$ (this is a map, not function).
        - Solution: transform $x$ into a feature vector $h \in \mathbb{R}^{l \times d}$
        - split $h$ into $l$ vectors $h_1, h_2, \dots h_l$
            
            ![image.png](VQVAE/image%202.png)
            
        - Use running estimates of $K$ cluster centroids over $h_{ij}$ and quantize $h_j$ into $h'_j$ using its nearest centroid $e_z$
            
            ![image.png](VQVAE/image%203.png)
            
        - Then concatenate $h_i' \dots h_l$ back into one vector $h'$ and transform $h'$ into $P(x | h’) = P(x | z_j))$.
        - The parameters that have to be learned are $\phi, \theta, \text{and}\  E = [e_1, e_2, \dots e_K]$ (the centroids of the clusters). It is best to start with an MLE approach to understand why modifications need to be made.
        - Recalling the ELBO for continuous VAEs:
            
            $$
            = -D_{KL}(Q_{\phi}(z | x) || P_{\theta}(z) + \mathbb{E}_{Q_{\phi}}[\log{P_{\theta}(x | z)}] 
            $$
            
        - For VQVAEs, you want $P(z) = Unif(1, \dots K) = \frac{1}{k} \forall z$
            - There is a deterministic encoder, i.e.
                
                $$
                Q(z|x) = \begin{cases}1 \ \text{if}\  z = \argmin_{k} || x - e_k||^2\\ 
                0 \ \text{else} 
                \end{cases}
                $$
                
            - By definition of KL Divergence,
            
            $$
            D_{KL} = Q_{\phi}(z|x) || P_{\theta}(z)) = \sum_{i=1}^KQ(z|x)\log\frac{Q(z|x)}{P(z)} = 1 \log \frac{1}{\frac{1}{K}} + \sum 0 \log\frac{0}{\frac{1}{K}} = \log K \ \ \ \text{since 0 log 0 = 0}
            $$
            
        - Since $\log K$ does not depend on a single one of the VQVAE parameters, it is exempt from the ELBO.
        - Second half of the ELBO:
            - like in a VAE, you could try sampling from $Q(z|x)$ and compute the log prob of the decoder ($\log P(x | z)$). or $-||x - \mu_x||^2$.
            - Same issue as with there through, we don’t know $P(x|z)$ and can’t find it in any reasonable way since we don’t know $P(x)$.
            - Unlike with VAEs, there is no reparamaterization trick, since we aren’t trying to output a distribution but rather have a discrete representation.
            - The main issue is that in backprop, the path to $\phi$ back is blocked due to non-differentiability.
                
                $$
                \frac{\partial \log P(x|z)}{\partial h '}\frac{\partial h'}{\partial h}\frac{\partial h}{\partial \phi}
                $$
                
            - Part of $\frac{\partial h’}{\partial h}$ is $E$, which is discrete and non differentiable.
            - As such, a custom loss function is used to maximize $E_{Q_\phi}[P(x | z)]$
            - You want each $h'$ to output a high $\log P(x | z)$ ($||x - \mu_x||^2$)
            - You also want  ($e_i \dots e_K$) to be the centroids of the K clusters among the $h_{ij}$ latent representations ($||h - e_z||^2$)
            - And you want each $h_i$ to be close to its nearest centroid (minimize SSD)
            - $\mathcal{L}(\phi, \theta, E) = ||x - \mu_x||^2 + ||h - e_z||^2$
            - You can express the gradient w.r.t $\phi$ as a product of jacobians
            - The only issue that remains is the nondifferentiability of the map from $h$ to $h'$.
            - The solution: **Ignore it lol**
            - They literally set it to $I$ in the paper.
                
                $$
                \frac{\partial \log P(x|z)}{\partial h '}I\frac{\partial h}{\partial \phi}
                $$
                
            - 2nd loss term still has non-zero gradient w.r.t $h$ and $e_z$
            - For more ease of use, the 2nd term is split into 2 with a stop gradient for separate calcs.
                
                $$
                \mathcal{L}(\phi, \theta, E) = ||x - \mu_x||^2 + ||h - sg(e_z||^2) + ||sg(h) - e_z||^2
                $$
                
            - If you sum them, you get the same expression, but separating them allows you to use an exponential moving average like in diffusion models or weight each loss term separately.