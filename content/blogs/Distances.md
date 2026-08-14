# On Distance and Flows: Steps, Chords, and Transport (WIP!)

![](../../assets/image-1.png){width=1080, height=720}

Recently, I have been going deeper and deeper into my niche of Generative Modeling. A not so old adage (source: Me) claims that when you start talking about Optimal Transport, it is either a sign of dementia or very deep AI psychosis. While I hope I am not (fully) demented, and I have not yet succumbed to the claude schizophrenia mind virus, there is likely some truth to the statement. To take a couple of steps back, an introduction to Optimal Transport is in order.

## What is Optimal Transport?

Optimal Transport is, bluntly, mathematically very beautiful. In order to understand the definition, you first have to understand distance. Like my prior post, I will harken back to high school education, even though this pissed me off to no end when someone did this to me when I was in a stochastic processes class. (No, I don't get why the drift coefficient follows these rules and converges to $\sqrt{\sigma}$ "just like high school calculus"). I will attempt to be a little gentler. 

If you recall learning the distance formula between two points, perhaps when being introduced to $y=mx+b$, you recall that this is used in triangles. Keep this triangle in mind.

If you stretch your memory a bit more, you can potentially recall one of your years in university, where you (potentially) took a linear algebra class, and learned that this is actually called the distance in $L_1$, norm or the magnitude between 2 vectors in euclidean space. That is all fine and dandy, but how does this connect to the near-incomprehensible word salad in the beginning of this article? Well, $\textbf{distance}$ is actually (like most things in math) ambiguously defined but the two things you definitely want, but don't always have are symmetry (i.e. no matter what order you put variables into a function, its value is the same) and the triangle inequality. 

## The Triangle Inequality

I promised triangles, and here it is. The triangle inequality is pretty simple. Effectively, it states that for any triangle, the sum of 2 lengths of any 2 sides (can be an isosceles, as you will see) must be greater than or equal to the length of the last side. Concretely, defining $a, b, c$ as side lengths, and c being the odd one out, $c \leq a + b$. 
That is very nice, and I am sure there is exactly 1 person who cares. But, in euclidean geometry, this is actually a theorem about vector norms. Mainly, the $L_1$ norm of the sum of 2 vectors is less than or equal to the sum of the magnitde of them by themselves. Being pedantic, the triangle inequality from the triangle still holds if the vectors are $\in \mathbb{R}^1$ and both vectors are real numbers. 

## How is this remotely useful?

Very good question. Not immediately apparent. However, a common problem in machine learning is to figure out how "close" two different data distributions are. Or, potentially, if you want to measure probability distributions. You can't --reasonably-- take every single point in distribution 1 and take the euclidean distance to a paired point in distribution 2. There exist measures of distance between probability distributions, but they do not make mathematicians happy. You might be familiar with the Kullback-Liebler divergence, especially if you come from a reinforcement learning background. This is a $\textbf{Divergence}$, meaning it does not satisfy both the property of symmetry and the triangle inequality (in the case of KL, the former is unsatified). For reference, $$D_{KL}(P\|Q) = \int p(x) \log \biggl(\frac{p(x)}{q(x)}\biggr)dx$$
and $P, Q$ are probability distributions being compared. 
KL is very handy (especially in generative modeling) because it has a very nice form between two gaussians, one which has diagonal covariance and one which has mean 0, and identity covariance: 
 
$$D_{KL}(\mathcal{N}(\mu_1, \dots, \mu_k)^T , \text{diag}(\sigma_1^2, \dots, \sigma_k^2)) \mathcal{N}(0,I) = \frac{1}{2}\sum_{i=1}^k[\sigma_i^2 + \mu_i^2 - 1 = ln(\sigma_i^2)]$$
There are a couple of other special cases, but I digress.


Here are couple issue with KL (Note that I say a couple, not all since there are a few, even though it is very helpful).  As I mentioned, KL is not symmetric. $D_{KL}(P\|Q) \neq D_{KL} (Q \|P)$. Techincally, you can make it symmetric with things like solving a geodesic equation, but that's beyond the scope of this. If you are interested, [this](https://people.csail.mit.edu/fisher/publications/papers/principe00vlsi.pdf) is worth a read. The greater issue is that the KL between $P$ and $Q$ can actually be almost infinity if the support of $P$ and $Q$ is not equivalent. There is a REALLY good blog I used for research/refreshing on OT that has a good graphic for this, and I will shamelessly take it but please visit it and read since Mr. Williams explains these topics far better than I can. 
![image](https://alexhwilliams.info/itsneuronalblog/code/ot/schematic_1d.png){width=720 height=480}

<div style="text-align: center"> [Alex Williams' Blog on Optimal Transport](https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/)  </div>

To summarize his image, and conclusion, there exist many distributions where KL says they are infinitely far apart $$D_{KL}(P \| Q) = D_{KL}(Q\| P) = + \infty$$
In his work, he says it better than I ever could. "Intuitively, some of these distribution pairs seem “closer” to each other than others. But the KL divergence says that they are all infinitely far apart. One way of circumventing this is to smooth (i.e. add blur to) the distributions before computing the KL divergence, so that the support of and matches." I mean just look at them! Are you going to tell me that in the third one $P$ and $Q$ don't look closer than in the second?

## The case for Wasserstein
So, clearly a divergence doesn't work. This instantly rules out Jensen-Shannon Divergence --technically a special case of KL but I digress--. Other explainations would require me to start going into the weeds on the Radon-Nikodym Derivative and other boring measure theory stuff, which while fun is probably orthogonal to both any reader's and my time. Good thing people about 50 times smarter than me came up with a real distance metric to measure probability distributions! This metric is called Earth Mover's distance, or more formally (and how I will refer to it) as Wasserstein Distance. Even more formally, we could call it Kantorovich-Rubenstein but that is annoying to type so I will use Wasserstein. Why Earth Mover's? Well, this requires understanding how Wasserstein works.

## How does Wasserstein work?
Assume that there is some amount of earth or dirt, representing $P$ and $Q$. There is a metric space $\mathcal{M}$ where each pile of earth is dropped on. The metric is how much effort, or the cost, of moving dirt from $P$ to $Q$. This is supposed to be the amount of dirt multiplied by the average distance you need to move the dirt. 

This is the part where, when I started learning about it, I got very confused until I realized it was invented by physicists, who never had to dig a hole in their lives. Normally, you do not really move dirt from a very nicely probablity shaped hole to another, but perhaps France has very Gaussian shaped holes. 

French soil notwithstanding, the actual important points are that there exists a metric which can help measure distance between two probability distributions, and (through roundabout mental gymnastics and proofs that would use up all of my github actions budget) fulfill all of the requirements to be called an actual distance metric.
I will circle back to Wasserstein, but it will be easier to work with if we finally discuss Optimal Transport

## Back to Optimal Transport

Since the understanding you currently have of Wassertein distance is that it's moving dirt between two holes, and that's about it, defining Optimal Transport is very helpful to providing some intuition. This is very hit or miss, and I'm not a fan or "you either get it or you don't", so I will attempt to provide alternative analogies and explainations. 


To start off with, a more formal definition is in order. For a distribution of mass $A(x)$ on a space $X$, you want to move that mass so it is transformed into the distribution $B(y)$ on $X$. This is the formal explaination of the "moving a pile of dirt from hole A to hole B". However, this analogy really only works if the pile of dirt you are about to make has the same mass as the dirt you are digging up from the first hole. This creates a key assumption for Optimal transport, that being that both $A$ and $B$ are distributions which have a mass of 1.

Now, putting ourselves into the shoes of the physicists who cannot fathom doing work and instead need to model it, we can define a cost function $c(x,y) \geq 0$ which is used to represent the cost of moving mass (I will stop using the dirt analogy for now, but it still holds) from the point $x$ to the point $y$. 

Physicists, notoriously disorganized people are also quite fond of plans. As such, this convoluted analogy also involves a plan of $\textbf{transport}$ to move $A$ into $B$ and can be described by a function $\gamma(x,y)$. This function will tell the frazzled, executive-function lacking physicist how much mass to move from point $x$ to $y$. 

For fear of falling into the same trap I have been lambasting physicists for, I will attempt to tie this in to the dirt analogy again. You can imagine the transport plan as moving soil shaped like $A$ to a hole in the ground shaped like $B$. At the end of this process, the pile of dirt $A$ and the hole $B$ should be completely gone since they have been fit as all of their mass has been transferred. 

Obviously, this explains why we need their masses to be 1. However, this also needs a couple of other properties

1. You can't move more dirt out of the hole than there was in the first place:
    $\int\gamma(x,y)dx = A(x)$

2. The amount of dirt you put into the hole has to be equal to the depth of the hole that was there when the lazy physicist started tossing dirt into it:
    $\int \gamma(x,y)dy = B(y)


This is kind of obvious if you have two convenient holes and piles of dirt in front of you, but I am assuming you don't. More plainly:

The mass moved out of a region around $x$ has to be equal to $A(x)dx$ and the mass moved into a region around $y$ has to be $B(y)dy$

Rewriting this, we get that , actually, $\gamma$ can be a joint probability distribution with marginals $A$ and $B$. So mass transport from $x$ to $y$ can be modeled by the equation $\gamma(x,y)dxdy$, with cost function $c(x,y)\gamma(x,y)dxdy$

Using the definition of joint probability, the total cost is the joint pdf 
$$\int \int c(x,y)\gamma(x,y)dxdy = \int c(x,y)d\gamma(x,y)$$

$\gamma$ (the plan) does not have a unique solution. The physicist is wont to get lost, or get his 5th white monster of the day. Like defined, for $\gamma$ to work it has to be a joint distribution with marginals $A, B$. The wikipedia article I stole some (most) of this notation and math from denotes the set of all "measures" as $\Gamma$, so I will do the same. 

The optimal plan, to deviate from the wikipedia article a bit, has a cost $$C = \inf_{\gamma \in \Gamma(A,B)}\int c(x,y)d\gamma(x,y)$$ 

In order for the math to be "cleaner", I will switch to measure notation $\mu, nu$ in place of $A, B$. I promise we are still referring to the same proverbial holes. 

Using the new notation, the set of allowed $\textbf{couplings}$, or pairs on this space is
$$
\Pi(\mu,\nu)
=\Bigl\{ \gamma \text{ on }X\times Y:
(\pi_X)_{\#}\gamma=\mu,\,
(\pi_Y)_{\#}\gamma=\nu
\Bigr\}.
$$

Unless I have reached a very specific audience, I am sure this notation is very confusing. 

Breaking it down, $\pi_X(x,y)=x$ and $\pi_Y(x,y)=y$ refer to what are called $projections$.
This is different from the linear algebra notion of projection, and is (in my opinion) easier to think about. Are mathematicians uncreative? Maybe. That aside, this definition of projection is really easy. 

For the example, assume have a point in the product space $(x,y) \in X \times Y$
Projection here means you just remove one of the coordinates from the coordinate system!

So, the projection of $Y$, for example ($\pi_Y$) would be $pi_Y(x,y) = X$. This is not a joke, and I am not sure why this needs to be quantified with a symbol, but I figured it was worth mentioning to avoid confusion.

As for the octothorpe. This is called the pushforward symbol. This means to only look at the $x$ coordinate of samples drawn from the plan $\gamma$

Converting back to PDF, 
the two marginals are now in the form of $X,Y$ and $A,B$: 
$$
\int_Y \gamma(x,y)\,dy=A(x),\qquad
\int_X \gamma(x,y)\,dx=B(y).
$$

The optimal solution to this is finally what is called the Kantorovich Optimal Transport Problem:

$$
\gamma^\star
\in
\operatorname*{arg\,min}_{\gamma\in\Pi(\mu,\nu)}
\int_{X\times Y} c(x,y)\,d\gamma(x,y).
$$

The optimizer is still a plan, just like the other $\gamma$s. It holds the same capabilities as other plans, meaning it can move mass from one distribution to another across points. There is a stricter framing of the OT problem, called the Monge Optimal Transport problem, but the restriction actually makes it far less robust for most applications (not all, as we will see.) [1,2]

TO WRITE:

## Monge vs. Kantorovich

## Optimal Transport in Discrete Space

## Sinkhorn Scaling

## Barycentric 

## Ok, finally back to Wasserstein Distance

## The Physicist's Idea of Transpor: Aimless Walking

## Adding a little bit of chaos never hurt, right?

## Dynamic OT




## Diffusion, Super Quick

## So, what are we supposed to predict?

## Yes, I'm bringing up SDEs again. Take your sensitive ass back to Model Welfare

## DDIM

## CFG and Prompt-Difference Fields

## Why no one does "1-Step Sampling"

## Latent Diffusion


## Flow Matching

## Conditional Regression

## Adding a stick helps models to fetch too

## Rectified Flows

## Stochastic Interpolants
- Boffi, Albergo
- My Work?


## Velocity fields and scores "fall out" of integration

## 1 path for density yields infinite potential SDEs

## Architectural Approaches for Modeling Latent Dynamics
- DiT, SiT

## MMDit

## Learning to Edit
- SDEdit
- Attention Control
- Inversion Based
- Numerical Solvers

## Getting from Dynamic OT to the Chord Edit

## Don't forget where you came from
- BVT (and eps)

## World Models...

- Latent Dynamics should have set an alarm bell off
- Energy Based Models ...
- Here's a secret. I do NOT care about images, but a picture is worth $\approx$ 14-16 words...




## References
[1] Filippo Santambrogio. *Introduction to Optimal Transport Theory*. arXiv:1009.3856, 2010. https://arxiv.org/abs/1009.3856

[2] Jean-David Benamou and Yann Brenier. “A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem.” *Numerische Mathematik* 84, 375–393, 2000. https://doi.org/10.1007/s002110050002

[3] Marco Cuturi. “Sinkhorn Distances: Lightspeed Computation of Optimal Transport.” NeurIPS, 2013. https://arxiv.org/abs/1306.0895

[4] Martin Arjovsky, Soumith Chintala, and Léon Bottou. “Wasserstein GAN.” ICML, 2017. https://arxiv.org/abs/1701.07875

[5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” NeurIPS, 2020. https://arxiv.org/abs/2006.11239

[6] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. “Score-Based Generative Modeling through Stochastic Differential Equations.” ICLR, 2021. https://arxiv.org/abs/2011.13456

[7] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. “High-Resolution Image Synthesis with Latent Diffusion Models.” CVPR, 2022. https://arxiv.org/abs/2112.10752

[8] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. “Flow Matching for Generative Modeling.” ICLR, 2023. https://arxiv.org/abs/2210.02747

[9] Xingchao Liu, Chengyue Gong, and Qiang Liu. “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.” ICLR, 2023. https://arxiv.org/abs/2209.03003

[10] Michael S. Albergo and Eric Vanden-Eijnden. “Building Normalizing Flows with Stochastic Interpolants.” ICLR, 2023. https://arxiv.org/abs/2209.15571

[11] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.” arXiv:2303.08797; revised 2025. https://arxiv.org/abs/2303.08797

[12] William Peebles and Saining Xie. “Scalable Diffusion Models with Transformers.” ICCV, 2023. https://arxiv.org/abs/2212.09748

[13] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. “SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers.” ECCV, 2024. https://arxiv.org/abs/2401.08740

[14] Liangsi Lu, Xuhang Chen, Minzhe Guo, Shichu Li, Jingchao Wang, and Yang Shi. “ChordEdit: One-Step Low-Energy Transport for Image Editing.” arXiv:2602.19083v2, 2026. https://arxiv.org/abs/2602.19083

[15] Minghan Li, Jeremy Moebel, and Mengyu Wang. “Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights.” arXiv:2606.14042, 2026. https://arxiv.org/abs/2606.14042

[16] ChordEdit authors. *Official ChordEdit implementation*. https://github.com/ChordEdit/ChordEdit

[17] Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising Diffusion Implicit Models.” ICLR, 2021. https://arxiv.org/abs/2010.02502

[18] Christian Léonard. “A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport.” *Discrete and Continuous Dynamical Systems A* 34(4), 2014. https://arxiv.org/abs/1308.0215

[19] Cédric Villani. *Optimal Transport: Old and New*. Springer, 2009. https://doi.org/10.1007/978-3-540-71050-9

[20] Alex H. Williams. “Optimal Transport: A Gentle Introduction.” 2020. https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/

[21] José C. Príncipe, Dongxin Xu, and John W. Fisher III. “Information Theoretic Learning.” Background link used in the original draft: https://people.csail.mit.edu/fisher/publications/papers/principe00vlsi.pdf

[22] ChordEdit project page. https://chordedit.github.io/

[23] Jonathan Ho and Tim Salimans. “Classifier-Free Diffusion Guidance.” arXiv:2207.12598, 2022. https://arxiv.org/abs/2207.12598

[24] Patrick Esser, Sumith Kulal, Andreas Blattmann, et al. “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.” ICML, 2024. https://arxiv.org/abs/2403.03206

[25] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. “An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale.” ICLR, 2021. https://arxiv.org/abs/2010.11929

[26] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. “Neural Ordinary Differential Equations.” NeurIPS, 2018. https://arxiv.org/abs/1806.07366

[27] Alexander Tong, Kilian Fatras, Nikolay Malkin, et al. “Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.” TMLR, 2024. https://arxiv.org/abs/2302.00482

[28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. “Elucidating the Design Space of Diffusion-Based Generative Models.” NeurIPS, 2022. https://arxiv.org/abs/2206.00364

[29] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. “DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.” NeurIPS, 2022. https://arxiv.org/abs/2206.00927

[30] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. “Consistency Models.” ICML, 2023. https://arxiv.org/abs/2303.01469

[31] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. “SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.” ICLR, 2022. https://arxiv.org/abs/2108.01073

[32] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. “Prompt-to-Prompt Image Editing with Cross-Attention Control.” arXiv:2208.01626, 2022. https://arxiv.org/abs/2208.01626

[33] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. “Adversarial Diffusion Distillation.” arXiv:2311.17042, 2023. https://arxiv.org/abs/2311.17042

[34] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. “Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference.” arXiv:2310.04378, 2023. https://arxiv.org/abs/2310.04378

[35] Aapo Hyvärinen. “Estimation of Non-Normalized Statistical Models by Score Matching.” *Journal of Machine Learning Research* 6, 695–709, 2005. https://jmlr.org/papers/v6/hyvarinen05a.html

[36] Yang Song and Stefano Ermon. “Generative Modeling by Estimating Gradients of the Data Distribution.” NeurIPS, 2019. https://arxiv.org/abs/1907.05600

[37] Brian D. O. Anderson. “Reverse-Time Diffusion Equation Models.” *Stochastic Processes and Their Applications* 12(3), 313–326, 1982. https://doi.org/10.1016/0304-4149(82)90051-5

[38] Pascal Vincent. “A Connection Between Score Matching and Denoising Autoencoders.” *Neural Computation* 23(7), 1661–1674, 2011. https://doi.org/10.1162/NECO_a_00142

[39] Tim Salimans and Jonathan Ho. “Progressive Distillation for Fast Sampling of Diffusion Models.” ICLR, 2022. https://arxiv.org/abs/2202.00512
