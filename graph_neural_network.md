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
