a = list(map(int, input().split()))

# 歩数データを7日単位に分割
chunks = [a[i:i + 7] for i in range(0, len(a), 7)]

# 各週間の歩数合計を計算
result = [sum(chunk) for chunk in chunks]

print(*result)
