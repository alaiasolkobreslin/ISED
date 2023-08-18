# neuro-symbolic-dataset

## Notes:
- [Drive Folder](https://drive.google.com/drive/folders/1e_Gm-ZNdAPsc64K1c5cQadUU7oaZqtOw?usp=sharing) containing Coffee leaf masks and other datasets
- Sudoku puzzles come from [here](https://github.com/alaiasolkobreslin/sudoku-puzzles)
  * For mini sudoku, min number of blanks is 6, max is 12
  * For regular sudoku, min number of blanks is 16, max is 52
- Coffee leaf dataset comes from this [Kaggle dataset](https://www.kaggle.com/datasets/alvarole/coffee-leaves-disease)
- SAM [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

TODO:
- Coffee leaf task
  * Clean up dataset by removing masks that are overlaping.
  * Fix severity score calculation to calculate ratio of rust to total leaf area.
- Sudoku
  * ~~Fix dataset to stop blanks/MNIST 0s from ever being forwarded through the CNN.~~
  * Run at high sample count for many epochs.
  * ~~Replace current solver with a SAT-based solver to improve speed.~~
  * ~~Find a dataset with a proper license or generate our own.~~
  * Question: is it ok for the solver function to return the input board (with no blanks filled in) when there is no solution? This is what I have implemented and I believe that it helps when we use similarity checking in the loss.
- CoNLL2003
  * Finish implementing this.
- Other
  * ~~Implement caching~~
  * ~~Add timeout back~~
  * Add baselines
