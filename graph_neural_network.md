# Graph neural network
- spectral approach : depend on a specific graph structure.
- non-spectral approach : define convolutions directly on the graph, operating on groups of spatially close neighbors.
  - GraphSAGE : a graph representaiton in inductive wayl

# Stanford Graph Learning Workshop
https://snap.stanford.edu/graphlearning-workshop/index.html
# CS224W: Machine Learning with Graphs
http://web.stanford.edu/class/cs224w/
# PyG 2.0: Advanced Representation Learning on Graphs | Fey | Computer Forum 2021 | [link](https://www.youtube.com/watch?v=oqHzTwzlWeQ&ab_channel=StanfordComputerForum)
The properties of graph:
- No fixed node ordering or reference point
- Often dynamic and have multimoda
- Multimodal node and edge features

## PyG (PyTorch Geometric)
```
from torch_geometric.nn impofr GCNConv

class GNN(torch.nn.Module):
  def __init__(self):
    self.conv1 = GNVConv
```

# Ref 
https://www.microsoft.com/en-us/research/video/msr-cambridge-lecture-series-an-introduction-to-graph-neural-networks-models-and-applications/  
TF Graph Neural Network Samples   
https://github.com/microsoft/tf-gnn-samples#brockschmidt-2019  
  
  
Node Classification with Graph Neural Network  
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/graph/ipynb/gnn_citations.ipynb#scrollTo=oRBphsN3whlI  

# Q&A
- Interpretation of Symmetric Normalised Graph Adjacency Matrix
  Ref: https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
- 
