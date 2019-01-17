# 按照惯例的实现

# 定义网络参数
def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2x3
  network['b1'] = [0.1, 0.2, 0.3] # 3,
  
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3x2
  network['b2'] = [0.1, 0.2] # 2,
  
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2x2
  network['b3'] = [0.1, 0.2] # 2,
  
  return network

def forward(network, x):
  W1,W2,W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'],network['b2'],network['b3']
  
  a1 = np.dot(x,W1) + b1 # 线性
  z1 = sigmoid(a1)
  
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3)
  
  return y # 最后的输出

# 实际调用
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708 0.69627909]

# 原文：https://blog.csdn.net/u011240016/article/details/85121601 
