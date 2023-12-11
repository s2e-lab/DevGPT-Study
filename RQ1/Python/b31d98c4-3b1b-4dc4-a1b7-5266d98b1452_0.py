def find_calculation(nums, target):
    # 与えられた2つの数aとbに対するすべての四則演算の結果を返す
    def operations(a, b):
        return [('+', a+b), ('-', a-b), ('*', a*b), ('/', a/b if b != 0 else None)]

    # 再帰関数
    def helper(nums):
        if len(nums) == 1:
            if nums[0] == target:
                return [(True, [])]
            return [(False, [])]
        
        results = []
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    for op, new_num in operations(nums[i], nums[j]):
                        if new_num is None:  # 0での除算を回避
                            continue
                        new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
                        new_nums.append(new_num)
                        for success, prev_ops in helper(new_nums):
                            if success:
                                results.append((True, prev_ops + [(nums[i], op, nums[j], new_num)]))
        return results

    # 実行
    calculations = helper(nums)
    return [ops for success, ops in calculations if success]

# テスト
nums = [2,4,5,8]
target = 10
for ops in find_calculation(nums, target):
    for a, op, b, result in ops:
        print(f"{a} {op} {b} = {result}")
    print("===")
