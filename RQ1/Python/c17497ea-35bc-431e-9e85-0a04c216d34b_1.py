import numpy as np
import matplotlib.pyplot as plt

# 設定x軸的範圍，通常設定一個週期的範圍即可
x = np.linspace(0, 2*np.pi, 100)

# 計算sin函數的值
y = np.sin(x)

# 繪製sin函數圖形
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sin 函數圖形')
plt.grid(True)
plt.show()
