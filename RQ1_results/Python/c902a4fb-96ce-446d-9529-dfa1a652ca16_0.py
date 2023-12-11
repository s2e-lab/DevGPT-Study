def test_Zero_One_Knapsack():
    test_cases = [
        ((5, [2, 3, 4], [3, 4, 5]), 7),
        ((7, [2, 3, 3, 4], [3, 4, 5, 6]), 11),
        ((10, [5, 5, 5, 5], [10, 20, 30, 40]), 70),
        ((8, [2, 3, 4, 5], [5, 5, 5, 5]), 10),
        ((3, [2], [4]), 4)
    ]
    
    for i, ((W, wt, vals), expected) in enumerate(test_cases):
        result = Zero_One_Knapsack(len(vals), wt, vals, W)
        assert result == expected, f"Test case {i} failed: expected {expected}, got {result}"
        print(f"Test case {i} passed")

test_Zero_One_Knapsack()
