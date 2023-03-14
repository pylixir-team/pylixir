def is_equal_in_fp_precision(a: list[float], b: list[float], err: float = 1e-6) -> bool:
    total_err = sum(abs(a_value - b_value) for a_value, b_value in zip(a, b))
    return total_err < err
