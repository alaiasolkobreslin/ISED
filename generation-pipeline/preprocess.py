
class Preprocess:

    def preprocess(input):
        pass


class PreprocessIdentity(Preprocess):

    def preprocess(input):
        return input


class PreprocessSort(Preprocess):

    def preprocess(input):
        return sorted(input)


class PreprocessSudokuBoard(Preprocess):

    def preprocess(input):
        # length = len(input)
        # TODO: make sudoku board
        return input
