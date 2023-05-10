from blackbox import *

def test_sum_2():
    sum_2 = BlackBoxFunction(
        lambda x, y: x + y,
        (DiscreteInputMapping(list(range(10))), DiscreteInputMapping(list(range(10)))),
        DiscreteOutputMapping(list(range(19))),
        sample_count=1)

    digit_1_distrs = torch.nn.functional.softmax(torch.randn(64, 10), dim=1)
    digit_2_distrs = torch.nn.functional.softmax(torch.randn(64, 10), dim=1)

    result = sum_2(digit_1_distrs, digit_2_distrs)

    print(result)
    print(result.shape)


def test_failable_sum_2():
    def f(x, y):
        if x == 0 and y == 0:
            raise Exception("BAD")
        else:
            return x + y

    sum_2 = BlackBoxFunction(
        f,
        (DiscreteInputMapping(list(range(10))), DiscreteInputMapping(list(range(10)))),
        DiscreteOutputMapping(list(range(19))),
        sample_count=1)

    digit_1_distrs = torch.nn.functional.softmax(torch.randn(64, 10), dim=1)
    digit_2_distrs = torch.nn.functional.softmax(torch.randn(64, 10), dim=1)

    result = sum_2(digit_1_distrs, digit_2_distrs)

    print(result)
    print(result.shape)


def test_hwf():
    def fn(symbols: List[str]):
        result = eval("".join(symbols))
        if abs(result) > 10000: raise Exception("BAD")
        else: return result

    hwf = BlackBoxFunction(
        fn,
        (ListInputMapping(7, DiscreteInputMapping([str(i) for i in range(10)] + ["+", "-", "*", "/"])),),
        UnknownDiscreteOutputMapping(),
        sample_count=100)

    symbols = torch.nn.functional.softmax(torch.randn(64, 7, 14), dim=2)
    lengths = [random.choice([1, 3, 5, 7]) for _ in range(64)]
    results, result_probs = hwf(ListInput(symbols, lengths))

    print(results, result_probs)


if __name__ == "__main__":
    # test_sum_2()
    # test_failable_sum_2()
    test_hwf()
