import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 2])

y = tf.transpose(x)

ans = x / 2

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(ans, feed_dict={x: [[1, 2]]}))
