from typing import Optional, List
import torch

EPS = 1E-6

from experiments.scene.dataset import classify_llm, SceneNet, scenes, objects
from experiments.scene import _SceneState
from experiments.scene.anesi_scene import ANeSIBase
from experiments.scene.im_scene import InferenceModelScene, IndependentIMScene

class SceneState(_SceneState):

    def len_y_encoding(self):
        return 0

class _SceneModel(ANeSIBase[_SceneState]):

    def __init__(self, args):
        self.N = args["N"]
        self.y_encoding = args["y_encoding"]
        self.model = args["model"]
    
        if self.model == "full":
            im = InferenceModelScene(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"],
                                    prune=args["prune"])
        elif args["model"] == "independent":
            im = IndependentIMScene(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"])
        super().__init__(im,
                         # Perception network
                         SceneNet(),
                         belief_size=[47]*10,
                         **args)

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        pass

    def initial_state(self, 
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None, 
                      generate_w=True) -> _SceneState:
        output_dims = self.output_dims(self.N, self.y_encoding)

        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(1)]
        y_list = None
        if y is not None:
            y_list = self.preprocess_y(y)
        return _SceneState(P, self.N, (y_list, w_list), output_dims, generate_w=generate_w)
    
    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        # output_dims = self.output_dims(self.N, self.y_encoding)
        return [y]
    
    def symbolic_function(self, w: torch.Tensor, box_len) -> torch.Tensor:
        return self.op(w, box_len)
    
    def op(self, x):
        raise NotImplementedError()
    
    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam and self.model == 'full':
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        prediction = prediction[0]
        return y == prediction

class SceneModel(_SceneModel):

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> SceneState:
        
        _initialstate = super().initial_state(P, y, w, generate_w)
        return SceneState(_initialstate.pw, _initialstate.N, _initialstate.constraint, _initialstate.y_dims, generate_w=generate_w)
    
    def op(self, xs, box_len) -> torch.Tensor:
      preds = []
      xs = xs.reshape(-1, len(box_len), 10)
      for x in xs:
        for n in range(len(box_len)):
            i = x[n, :box_len[n]]
            input = [objects[int(j)] for j in i]
            input.sort()
            y_pred = classify_llm(input)
            preds.append(torch.tensor(scenes.index(y_pred)))
      return torch.stack(preds).long()

    def output_dims(self, N: int, y_encoding: str):
      return [9]

