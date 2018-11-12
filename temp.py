import numpy as np

np.random.seed(123)
# w = np.random.randn(6)

y = np.rint(np.random.rand(4))
y = y.copy()

x = np.random.rand(24).reshape(4,6)
x = x.copy()/2

#np.dot(w.T,x) # same as np.dot(b,a)

 # shuffle the training set
        np.random.seed(100+epoch)
        np.random.shuffle(train)

w_gd = gd_weight(x,y,0.1,1,1)
w_sgd = sgd_weight(x,y,0.1,1,1)
print(w_gd)
# p0 = np.empty((0,5),float)
# # new = np.vstack((e,p))