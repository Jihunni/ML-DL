# Machine Learning Basics
- Machine learning sub-problems
  - classification : data to discrete class label. Predicting a class label
  - regression : predicting a numerical value
  - similarity : finding similar/dissimilar data
  - clustering : discovering structure in data
  - embedding : data to a vector
  - reinforcement learning : training by feedback
- Machine Learning Model Evaluation Metrics | PyData LA 2019 | [video](https://youtu.be/PeYQIyOyKB8)
  - classification error metrics
    - accuracy : low performance on unbalanced data
    - mean Average precision (mAP)
    - confusion matrix
    - F1 score
    - AUC
    - and ...
  - regression error metics
    -  R^2
    -  mean square error
    -  absolute error
    -  root mean squared logarithmic error
    -  mean absoloute percentage error
  - permutation invariant: a model that produces the same output regardless of the order of elements in the input vector
    e.g. permutation invariant model : MLP
    e.g. permutation invariant operation : sum, mean, median, max, min
    e.g. permutation variant model : CNN, RNN --> position information
  - 
# Solution for overfitting
- data augmentation
- ensemble
  Ref: https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/ 
- dropout
- skip connection 
  Ref: https://lswook.tistory.com/105  
  ensemble-like structure
  ![image](https://user-images.githubusercontent.com/48517782/142642903-49b4a5d6-75aa-472a-9900-a2b79f671a83.png)

# Optimization
- `paper` Taking the Human Out of the Loop: A Review of Bayesian Optimization

# GNN (Graph Neural Network)
- Intro to graph neural networks (ML Tech Talks) | [video](https://www.youtube.com/watch?v=8owQBFAHw7E)
  - application of GNN : prediction of drug properties from structure
  - A Deep Learning Approach to Antibiotic Discovery | Cell, 2020 | [paper](https://www.sciencedirect.com/science/article/pii/S0092867420301021)
  - application of GNN : estimate ETA
  - application of GNN : social network
## Graph Convolution
- spectral approahces : to redefine the convolution operation in the Fourier domain, utilizing spectral filters that use ghe graph Laplacian.
- non-spectral approaches : to define the colvution operation directly on the graph.
  - GraphSAGE : node embedding through sampling and aggregation
  - Graph Attention Network (GAT)
## Graph pooling
- topology based pooling : 
  - graph coarsening algorithms
- global pooling architecture : Node feature representation. effective on a graph with smaller number of nodes.
- hierarchical pooling architecture : effective on a graph with larger number of nodes.
  - DiffPool
  - graph pooling (gPool) 
  
 - Graph U-Net
- Self-Attention Graph pooling (SAGpool)

# GNN paper
- Modeling Polypharmacy Side Effects with Graph Convolutional Networks

# RNN
## RNN basics
- [[Korean blog] RNN basic](https://taeu.github.io/nlp/deeplearning-nlp-rnn)
- What is the output in a RNN? | [stack overflow](https://math.stackexchange.com/questions/3107394/what-is-the-output-in-a-rnn)
- return_sequences
  - `return_sequences=True` : [batch_size, time_steps, input_dimensionality(input_features)] (contining the output for all time steps)
  - `return_sequences=False` : [batch size, input_dimensionality(input_features)] (containing the output of the last time stamp)
- TimeDistributed(Dense) vs Dense in Keras - Same number of parameters | [stack overflow](https://stackoverflow.com/questions/44611006/timedistributeddense-vs-dense-in-keras-same-number-of-parameters/44616780)
  - `TimeDistributedDense` applies a same dense to every time step during GRU/LSTM Cell unrolling. So the error function will be between predicted label sequence and the actual label sequence. (Which is normally the requirement for sequence to sequence labeling problems).
  - with `return_sequences=False`, Dense layer is applied only once at the last cell. This is normally the case when RNNs are used for classification problem. If `return_sequences=True` then Dense layer is applied to every timestep just like TimeDistributedDense.

## LSTM
- learn what recognize an important input (input gate), store it in the long-term state, preserve it for as long as it is needed (forget gate), and extract it whenever it is needed.
- two hidden states: h_t (short-term state; 'h' stands for 'hidden'), c_t (long-term state; 'c' stands for 'cell')
- gate controller : the logistic activation function. Its output ranges from 0 to 1. Output 0 means closure of the gate. Output 1 means openness of the gate.
  - forget gate
  - input gate
  - output gate

## creating training dataset : window

# attention
## previous research
- layer normalization
- residual connection
  - ResNet
- 

## papers
- Neural Machine Translation by Jointly Learning to Align and Translate | [paper1](https://arxiv.org/abs/1409.0473) 
- the concept of attention mechanism | [video](https://youtu.be/6aouXD8WMVQ)
- transformer | [video](https://www.youtube.com/watch?v=AA621UofTUA)
## concepts
- query, key, value
- attention pooling : Given a query, attention pooling biases selection over values.
- attention scoring function : a weighted sum of the values based on these attention weights
  - masked softmax operation
  - additive attention
  - scaled dot-product attention
- Bahdanau Attention : encoder-decoder
  `paper` Neural Machine Translation by Jointly Learning to Align and Translate | [paper1](https://arxiv.org/abs/1409.0473) 
- self-attention (intra-attention) and positional encoding
  - self-attention : query, key, and values come from the same place
  - positional encoding :  to use the sequence order information, we can inject absolute or relative positional information by adding positional encoding to the input representation.
    X + P (X : input representation, P: positional embedding matrix)  
    
  - 
## transformer
- Attention Is All You Need
- What exactly are keys, queries, and values in attention mechanisms? | [stack overflow](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)
- Pay Attention to MLPs

# computer vision
- two stage detection : slow, accurate
  - R-CNN
  - fast R-CNN : 
  - faster R-CNN : regional proposal network
- one stage detection : fast, inaccurate, not easy to train  
  Frame per Second (FPS) > 30 : criteria for real time visualization  
  - YOLO : grid --> conditional class probility and bouding boxes + confidence --> final detection
  - Single Shot Detection (SSD)
- application
  - image colorization
## Generative adversarial network
- gaugan system
  - https://blogs.nvidia.com/blog/2019/03/18/gaugan-photorealistic-landscapes-nvidia-research/
- cycleGAN : cycle-consistant adversarial network
  - idea : A -> B -> A'
  - cycle-consistency loss : abs(A - A')
# NLP
## Allen Institute for AI  
Perspective : All thihg in ML/DL is human efforts. (e.g. training data selection, loss function, model architecture)  
Yejin Choi  
Q. how to reduce the sterotype or bias such as racism or sexism?  
  - dateset
    - "garbage in, garbage out"
    - data augmentation
  - objective function
    - a traditional objective function minimize/maximize the error
    - To handling the bias, add 'gender' variable and a constraint that make 'gender' variable equally.
## basic stuff
- `review` A Primer on Neural Network Models for Natural Language Processin
## distributed representation (word embeddings) : low-dimensional space
- word embeddings
- Word2vec
  - CBOW
  - skip-gram
- Chateracter Embedding : out-of-vocabulary (OOV) words
- Contextualized word embeddings
  - Embedding from Language Model (ELMo)
    Ref : https://wikidocs.net/33930

# Generative model
- Generative adversarial networks (GANs)
- autoregressive models
- flows
- variational autoencoders (VAEs)
  - `paper` Auto-Encoding Variational Bayes [link](https://arxiv.org/abs/1312.6114)
# MLOps
- https://madewithml.com/
- https://mlops-for-all.github.io/
- https://fullstackdeeplearning.com/
- http://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&ejkGb=KOR&barcode=9791158392888
- https://www.youtube.com/watch?v=8R4DDEqjc0I
- production kubernetes
- Kubernetes in Action
