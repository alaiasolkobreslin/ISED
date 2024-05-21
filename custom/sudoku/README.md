
# Visual Sudoku

## Requirements
1. Run script `setup.sh`.

2. Install [Prolog](https://www.swi-prolog.org/Download.html).
To check if the Prolog installation is succesfull try to call Prolog from terminal by typing "prolog" (to exit prolog type `halt.`)

    NOTE: if unable to install prolog we provide another symbolic-solver for Sudoku: "backtrack" (which is slower).

3. Run `python src/sudoku_data.py --solver default` to generate the data. With problems installing prolog use `--solver backtrack`.

4. Download [pre-trained models](https://github.com/SamsungLabs/NASR/tree/main/outputs) and put `mask`, `perception`, and `solvernn` folders under `custom/sudoku/outputs`. 

## Experiments
To reproduce the experiments from the paper, run
```bash
cd custom/sudoku
python src/<METHOD>.py
```