a = list(map(int, input().split()))
print(len(a))
# 各週の合計値を求める
result = [0] * len(a)
for i in range(len(a)):
    result[i] = sum(a[:i+1])

print(*result)
