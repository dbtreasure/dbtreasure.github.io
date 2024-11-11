---
layout: post
title: "The Neural Net Tech Tree"
date: 2024-11-11 14:00:00 -0700
categories: neural-nets tech-tree
---

As my first post I'm going to introduce myself briefly, explain my approach to learning more about deep learning, and introduce my first project.

## Who am I?

I'm a software engineer working for a company that builds on top of large language models. I got this job after doing some consulting work in 2023 for a few different companies building retrieval systems, text-to-SQL, and document analysis. Prior to that period I have done front end web development and iOS application development.

Large language models and deep learning make me feel like how I imagine engineers felt like in the 1970s when the microprocessor was becoming a thing, how engineers in the 1980s felt when the PC was becoming a thing, and how engineers felt in the 1990s when the internet was becoming a thing.

## Where do I want to go?

Deep learning is extremely cool and I want to understand it better. I want to build my way through the "tech tree" of deep learning.

### What's a tech tree?

![The Neural Net Tech Tree](/assets/images/diablo2-tech-tree.png)
_The Diablo 2 Barbarian Tech Tree_

The tech tree is a way to visualize the progression of a skill in a game. In Diablo 2 after picking a character you could study the tech trees and identify where you wanted to end up. Keeping your destination in mind made it easier to grind through difficulties.

I believe that laying out a tech tree for deep learning will help me get to where I want to go. I want to build my way through different skills in deep learning, starting at the bottom and working my way up.

### The Deep Learning Tech Tree

![The Deep Learning Tech Tree](/assets/images/UpdatedDeepLearningTechTree.png)

With the help of ChatGPT I have created a tech tree for deep learning. To master a node on the graph I plan on studying the literature of published papers related to the node, building an implementation of the technique, and then capping that node off by publishing a blog post about it.

### What's in the tree?

I haven't mastered these topics yet so here's ChatGPT's description of the tech tree:

_This deep learning tech tree charts the major models, techniques, and architectures in deep learning, organized roughly chronologically and by dependencies. It serves as a roadmap for learning deep learning concepts, beginning with foundational principles and advancing through increasingly complex and powerful models. Each node represents a key innovation that has pushed AI forward, making it an ideal learning path for anyone aspiring to become an AI engineer._

## Tier 1: Foundations

_• Perceptron (1957): The perceptron is the simplest type of artificial neuron and forms the foundation of deep learning. It introduces basic concepts like weights, biases, and activation functions._

_• Feedforward Neural Network (FNN): An expansion of the perceptron, FNNs are networks where information moves only in one direction—forward—from input to output. Learning FNNs provides a solid grounding in neural network structure and basic training methods like backpropagation._

## Tier 2: Convolutional Models

_• Convolutional Neural Network (CNN) (1989): CNNs add convolutional layers that are particularly good at recognizing patterns in images, making them foundational for computer vision._

_• AlexNet (2012): AlexNet is one of the first CNNs to demonstrate the potential of deep learning on large image datasets. It introduced key techniques like ReLU activation and dropout to reduce overfitting, which are now standard in deep networks._

_• VGG (2014): VGG improved upon AlexNet by stacking small convolutional filters, creating a deeper network with more layers that achieved state-of-the-art accuracy. VGG networks are straightforward and help reinforce the importance of depth in CNNs._

_• ResNet (2015): ResNet introduced residual connections, allowing networks to skip layers, which helps solve the problem of vanishing gradients in very deep networks. This architecture is essential for anyone looking to work with modern deep CNNs in vision tasks._

## Tier 3: Recurrent and Sequence Models

_• Recurrent Neural Network (RNN): Unlike CNNs, RNNs are designed for sequence data. They introduce feedback connections, allowing information to loop through the network, which is useful for sequential data like text or time series._

_• Long Short-Term Memory (LSTM) (1997): LSTMs improve on RNNs by introducing memory cells that help retain information across long sequences. This architecture addresses the problem of vanishing gradients in sequence data and remains a staple for sequential data processing._

_• Gated Recurrent Unit (GRU) (2014): A simpler variant of LSTMs, GRUs are computationally efficient and often work similarly to LSTMs in practice._

## Tier 4: Attention Mechanisms

_• Attention Mechanism (2014): Attention mechanisms revolutionized sequence models by allowing networks to focus on the most relevant parts of an input sequence. They paved the way for highly effective models in NLP, like transformers._

_• Self-Attention (2017): This is a specific form of attention where each element of a sequence attends to every other element, enabling a model to capture long-range dependencies. Self-attention is crucial in transformer-based models._

## Tier 5: Transformers

_• Transformer (2017): Transformers use self-attention to process input sequences in parallel rather than sequentially, making them more efficient and scalable for large datasets. Transformers are the basis of most modern NLP models._

_• BERT (2018): BERT (Bidirectional Encoder Representations from Transformers) introduced bidirectional training for language models, allowing for a better understanding of context in NLP tasks._

_• GPT (2018): GPT models focus on unidirectional generation, predicting the next word in a sequence, which makes them effective for text generation._

_• Vision Transformer (ViT) (2020): ViTs adapt transformers to image data, marking a shift from CNN-dominated computer vision to transformer-based approaches._

## Top Tier: Advanced Transformers and Generative Models

_• GPT-3 (2020): GPT-3 demonstrates the potential of large-scale language models, with billions of parameters that allow it to generate high-quality text across many contexts._

_• DALL-E and CLIP (2021): DALL-E and CLIP combine vision and language understanding, training models to generate images from text prompts (DALL-E) and align image-text representations (CLIP)._

_• Score-Based Models (2020): Score-based models, including diffusion models, are designed for generating complex, high-quality data by iteratively refining noise._

_• Stable Diffusion (2022): Building on score-based models, Stable Diffusion is a powerful generative model that produces high-quality images from text descriptions. It combines techniques from diffusion models and multimodal transformers, showcasing the synthesis of ideas in modern AI._

## What's next?

Step one on my tree is the Perceptron. I'll pick out some papers, and books that I think will help me understand the topic. After that I'll build an implementation of the technique.

### In Karpathy I will trust

![Our Patron Saint of Deep Learning, Andrej Karpathy](/assets/images/for_respected_customer.png)
_Our Patron Saint of Deep Learning, Andrej Karpathy_

I plan on following in the [footsteps](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) of Andrej Karpathy when it comes to implementation. Start with something you understand in Python, if necessary add in PyTorch to make it easier, then to have some fun circle back and implement it over again in C, or Zig or [Autograd](https://docs.tinygrad.org/).

## If you don't publish it, it doesn't exist

An important part of this learning path is to publish what I'm learning. I don't care if I make my way down part of the tree and desist and turn back. Regardless of how "done" something is I will do a writeup on it. Writing exposes misunderstanding. It reveals what is unworthy of repetition. Writing _is_ learning.

## Summary

I've shared a bit about myself, my goals, and my approach. Next time I'll write about my first project: the Perceptron. See you then!
