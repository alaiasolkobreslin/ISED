
# Visual Sudoku
Use the pre-trained models in the `outputs` folder. 

## Installation
1- Run script `setup.sh` or follow its steps in order.

2- Install Prolog (instructions [here](https://www.swi-prolog.org/Download.html)).
To check if the Prolog installation is succesfull try to call Prolog from terminal by typing "prolog" (to exit prolog type `halt.`)
NOTE: if unable to install prolog we provide another symbolic-solver for Sudoku: "backtrack" (which is slower).

3- Run `python src/sudoku_data.py --solver default` to generate the data. With problems installing prolog use `--solver backtrack` (it might take longer to run). More options/statistics available in `src/sudoku_data.py`.

## Experiments
To reproduce the experiments from the paper, run
```bash
`cd custom/sudoku`
python src/<METHOD>.py
```