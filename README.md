# RNN
## RNN basics
- [[Korean blog] RNN basic](https://taeu.github.io/nlp/deeplearning-nlp-rnn)
- return_sequences
  - `return_sequences=True` : [batch_size, time_steps, input_dimensionality(input_features)] (contining the output for all time steps)
  - `return_sequences=False` : [batch size, input_dimensionality(input_features)] (containing the output of the last time stamp)
- TimeDistributed(Dense) vs Dense in Keras - Same number of parameters | [stack overflow](https://stackoverflow.com/questions/44611006/timedistributeddense-vs-dense-in-keras-same-number-of-parameters/44616780)
  - `TimeDistributedDense` applies a same dense to every time step during GRU/LSTM Cell unrolling. So the error function will be between predicted label sequence and the actual label sequence. (Which is normally the requirement for sequence to sequence labeling problems).
  - with `return_sequences=False`, Dense layer is applied only once at the last cell. This is normally the case when RNNs are used for classification problem. If `return_sequences=True` then Dense layer is applied to every timestep just like TimeDistributedDense.

## creating training dataset : window
