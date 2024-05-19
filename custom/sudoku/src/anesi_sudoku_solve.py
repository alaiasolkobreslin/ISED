from typing import List, Optional
import torch

from anesi_sudoku import ANeSIBase
from state_sudoku import _SudokuState
from im_sudoku import InferenceModelSudoku, IndependentIMSudoku

from sudoku_solver.board import Board

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')

EPS = 1E-6

class SudokuState(_SudokuState):

    def len_y_encoding(self):
        return 81

class _SudokuModel(ANeSIBase[_SudokuState]):

    def __init__(self, args):
        self.N = args["N"]
        self.y_encoding = args["y_encoding"]
        self.model = args["model"]

        if self.model == "full":
            im = InferenceModelSudoku(self.N,
                                      self.output_dims(self.N, self.y_encoding),
                                      layers=args["layers"],
                                      hidden_size=args["hidden_size"],
                                      prune=args["prune"])
        elif args["model"] == "independent":
            im = IndependentIMSudoku(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"])
        super().__init__(im,
                         # Perception network
                         belief_size=[[9]*81, [1]*81],
                         **args)

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        pass

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> _SudokuState:
        output_dims = self.output_dims(self.N, self.y_encoding)
        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(self.N * 81)]
        y_list = None
        if y is not None:
            y_list = self.preprocess_y(y)
        return _SudokuState(P, self.N, (y_list, w_list), output_dims, generate_w=generate_w)
    
    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        output_dims = self.output_dims(self.N, self.y_encoding)
        y_list = [y[:, i] for i in range(len(output_dims))]
        return y_list

    def symbolic_function(self, w: torch.Tensor, solution_boards_new) -> torch.Tensor:
        """
        w: (batch_size, 2*n)
        """
        return self.op(w, solution_boards_new)
    
    def op(self, n: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam and self.model == "full":
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        prediction = torch.stack(prediction).t()
        return prediction == y
    
class SudokuModel(_SudokuModel):

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> SudokuState:

        _initialstate = super().initial_state(P, y, w, generate_w)
        return SudokuState(_initialstate.pw, _initialstate.N, _initialstate.constraint, _initialstate.y_dims, generate_w=generate_w)
    
    def op(self, clean_boards, solution_boards_new) -> torch.Tensor:
        final_boards = []
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl") 
        for i in range(len(clean_boards)):
            board_to_solver = Board(clean_boards[i].reshape((9,9)).int().cpu())
            try:
                solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
            except StopIteration:
                solver_success = False
            final_solution = torch.from_numpy(board_to_solver.board.reshape(81,))
            if not solver_success:
                n = torch.randint(len(solution_boards_new), (1,))[0]
                final_solution = solution_boards_new[n].cpu()
            final_boards.append(final_solution)
        return torch.stack(final_boards)-1

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        return [9] * 81
