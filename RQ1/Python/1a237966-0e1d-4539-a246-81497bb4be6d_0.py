def find_pair_with_target_sum(nums, target):
    left = 0                   # Initialize the left pointer to the start of the array
    right = len(nums) - 1      # Initialize the right pointer to the end of the array
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:             # If the current sum equals the target
            return [left, right]              # Return the indices of the pair
        
        elif current_sum < target:            # If the current sum is less than the target
            left += 1                         # Move the left pointer to the right
            
        else:                                 # If the current sum is greater than the target
            right -= 1                        # Move the right pointer to the left
    
    return -1                                # No pair found, return -1

