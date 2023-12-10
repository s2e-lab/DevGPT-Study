# 假设 current_data 和 next_data 是相邻时间点的数据值
current_data = 1000
next_data = 890

# 计算数据差异的绝对值
diff = abs(current_data - next_data)

# 设置波动幅度的阈值
threshold = 10

# 如果数据差异超过阈值，则触发报警
if diff > threshold:
    print("报警：数据波动幅度超过阈值")
else:
    print("数据波动正常")
