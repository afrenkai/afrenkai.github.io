# Learning to detect AI-Generated Content with Diffusion Models

During my time at Pangram Labs, I have had the opportunity and agency to pursue a lot of what we call "moonshot" ideas. Most of them have not really panned out (due to my Shiny Object Syndrome), but one that did pan out were experiments with Discrete Diffusion Models: Masked and Uniform State Diffusion Language Models (M/USDLMs). 

## A Brief Background on RAID

A common benchmark used to evaluate the efficacy of an AI detection method is the (RAID)[linkhere] benchmark, created by Liam Dugan in 2024, with a separete track for COLING in 2025. Pangram already tops the COLING track, but has never been submitted to the main benchmark. This is both to ensure we do not corrupt our evaluation benchmarks with that of RAID's (so that we can be independently tested) and also because RAID has very old base models, which we do not target anymore (but encourage people to evaluate us on!). This is a product of the time RAID came out, but the field moves fast. Regardless, RAID is a very, very valuable benchmark and offers a plethora of insights into how AI detection is modeled in open source, how base models' latent signatures differ to that of those trained RLHF (reinforcement learning with human feedback) to follow instructions, and approaches that favor one or the other. 

## Why Diffusion?

Most of the high-performing approaches on RAID are some amalgamation of what is called a BERT, or Bidirectional Encoder Recurrent Transformer. The current highest performing approach, from Tencent, is completely closed source and as a result I can offer no insight into it. BERT operates by using a Transformer in both directions, i.e. sending the context of the model in the forward direction and then using another transformer to send it backwards, effectively repeating the context. This does effectively double the parameter count of the model, but allows for deeper representation and understanding of a sequence. BERT-esque models are better used in the modern day for things like late-stage interaction retrieval (COLBERT) and .... 

Diffusion comes into the picture because Masked (and, by extension, Uniform State) Diffusion Language Models operate on a similar principle.  They differ from BERT in that they do not predict the next token, but rather predict an approximation of a sequence. They are provided a sequence up to a point, and have to "unmask" in the case of a masked diffusion model the tokens moving backwards.[Add Picture]
This, along with some very helpful mathematical properties of Discrete Diffusion Models, allow for a deeper representation than naive BERT models. The difficulties lie in scaling these models, general mode collapse (ensuring that text remains coherent), and avoiding reward-hacking in the post-training phase. 

## So about that Post-Training Phase
In general, Reinforcement Learning for Diffusion Language Models is far different from that of Autoregressive Causal Language Models. Just this (2026) year, the best papers at ICML were all about this problem! 

