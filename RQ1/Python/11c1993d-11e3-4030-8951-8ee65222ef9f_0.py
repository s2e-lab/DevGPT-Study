def subarraySum(arr, start, end):
    n = len(arr)
    prefix = [0] * n  # Initialize the prefix sum array with zeros

    prefix[0] = arr[0]  # The first element of the prefix sum array is the same as the first element of the input array
    for i in range(1, n):
        prefix[i] = prefix[i-1] + arr[i]  # Calculate the prefix sum for each element of the input array

    if start == 0:
        return prefix[end]  # If 'start' is 0, the sum is simply prefix[end]
    else:
        return prefix[end] - prefix[start-1]  # Otherwise, subtract the prefix sum at index 'start-1' from prefix[end]
