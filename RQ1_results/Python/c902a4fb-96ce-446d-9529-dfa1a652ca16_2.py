def test_Longest_SS(self):
    # Test case 1
    X = "ABC"
    Y = "DEF"
    expected_output = ""
    self.assertEqual(Longest_SS(X, Y), expected_output)

    # Test case 2
    X = "ABC"
    Y = "BCD"
    expected_output = "B"
    self.assertEqual(Longest_SS(X, Y), expected_output)

    # Test case 3
    X = "ABCBDAB"
    Y = "BDCAB"
    expected_output = "BCAB"
    self.assertEqual(Longest_SS(X, Y), expected_output)

    # Test case 4
    X = "ABC"
    Y = "ABC"
    expected_output = "ABC"
    self.assertEqual(Longest_SS(X, Y), expected_output)

    # Test case 5
    X = "ABC"
    Y = "ABCD"
    expected_output = "ABC"
    self.assertEqual(Longest_SS(X, Y), expected_output)
