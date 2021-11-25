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
- Neural Machine Translation by Jointly Learning to Align and Translate | [paper1](https://arxiv.org/abs/1409.0473) 
- the concept of attention mechanism | [video](https://youtu.be/6aouXD8WMVQ)
- transformer | [video](https://www.youtube.com/watch?v=AA621UofTUA)

# transformer
- Attention Is All You Need
- What exactly are keys, queries, and values in attention mechanisms? | [stack overflow](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)
- Pay Attention to MLPs
