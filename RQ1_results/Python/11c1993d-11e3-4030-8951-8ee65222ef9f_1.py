def countSubarraysWithSum(arr, target_sum):
    prefix_sum = 0  # Initialize the prefix sum variable to zero
    sum_count = {0: 1}  # Initialize the dictionary with one entry for a prefix sum of zero (encountered before starting traversal)
    count = 0

    for num in arr:
        prefix_sum += num  # Update the prefix sum by adding the current element
        if prefix_sum - target_sum in sum_count:
            count += sum_count[prefix_sum - target_sum]  # If the prefix sum - target sum exists in the dictionary, increment the count
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1  # Increment the count of the current prefix sum in the dictionary

    return count
