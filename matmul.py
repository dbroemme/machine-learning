import numpy as np
import time

def l_mul(x, y):
    """
    Implements the L-Mul operation as described in the sources.

    Args:
        x: First floating-point operand.
        y: Second floating-point operand.

    Returns:
        The result of the L-Mul operation.
    """

    # Extract sign, exponent, and mantissa for both operands
    sign_x = x < 0
    sign_y = y < 0
    exponent_x = int(abs(x).hex().split('p')[1])
    exponent_y = int(abs(y).hex().split('p')[1])
    mantissa_x = abs(x) * 2**(-exponent_x) - 1
    mantissa_y = abs(y) * 2**(-exponent_y) - 1

    # Calculate the number of mantissa bits
    m = len(mantissa_x.hex().split('p')) - 2

    # Define the offset exponent l(m) based on the sources
    l_m = m if m <= 3 else (3 if m == 4 else 4)

    # Perform L-Mul operation
    result_mantissa = mantissa_x + mantissa_y + 2**(-l_m)
    result_exponent = exponent_x + exponent_y

    # Construct the result
    result = result_mantissa * 2**result_exponent

    # Apply the sign
    if sign_x != sign_y:  # XOR operation for sign
        result = -result

    return result

a = 2.982323
b = 3.235125
c = l_mul(a, b)

print(c)

print(a * b)

def run_timing_test(num_pairs=10000):
    # Generate random floating point numbers
    numbers = np.random.uniform(-100, 100, (num_pairs, 2))
    
    # Test l_mul
    start_time = time.perf_counter()
    for a, b in numbers:
        result = l_mul(a, b)
    l_mul_time = time.perf_counter() - start_time
    
    # Test native multiplication
    start_time = time.perf_counter()
    for a, b in numbers:
        result = a * b
    native_time = time.perf_counter() - start_time
    
    print(f"\nTiming Results ({num_pairs} pairs):")
    print(f"L-Mul time:    {l_mul_time:.6f} seconds")
    print(f"Native time:   {native_time:.6f} seconds")
    print(f"L-Mul is {l_mul_time/native_time:.2f}x slower than native multiplication")

# Run the timing test
run_timing_test()