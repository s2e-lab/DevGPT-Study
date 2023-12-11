# 入力リスト
input_list = [25, 50, 75, 100, 150]

# 全ての要素が25で割り切れるかどうかを判定
divisible_by_25 = all(num % 25 == 0 for num in input_list)

print(divisible_by_25)
