import tensorflow as tf

# 创建图
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
# 创建会话
sess = tf.Session()
# 计算 c
print(sess.run(c))  # 进行矩阵乘法，输出[3., 8.]
sess.close()
