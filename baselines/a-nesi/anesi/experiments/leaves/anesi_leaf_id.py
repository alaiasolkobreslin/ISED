import math
from typing import Optional, List

import torch

EPS = 1E-6

from experiments.leaves.anesi_leaf import LeafModel
from experiments.leaves.dataset import classify_11
from experiments.leaves import LeafState

class LeavesState(LeafState):

  def len_y_encoding(self):
    return 1
  
  
class LeavesModel(LeafModel):
  
    def initial_state(self,
                      P1: torch.Tensor,
                      P2: torch.Tensor,
                      P3: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> LeafState:
        
        _initialstate = super().initial_state(P1, P2, P3, y, w, generate_w)
        return LeafState(_initialstate.pw, _initialstate.N, _initialstate.constraint, _initialstate.y_dims, generate_w=generate_w)
    
    def op(self, margin: torch.Tensor, shape: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
      result = torch.ones(margin.shape)
      for i in range(len(margin)):
        m = l11_margin[margin[i]]
        s = l11_shape[shape[i]%5]
        t = l11_texture[texture[i]%4]
        y = classify_11(m, s, t)
        result[i] = l11_labels.index(y)
      return result.long()
    
    def output_dims(self, N: int, y_encoding: str):
      return [11]

l11_margin = ['entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate']
l11_shape = ['elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate']
l11_texture = ['glossy', 'leathery', 'medium', 'rough']
l11_labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']


