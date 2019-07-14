from keras.models import load_model
import numpy as np

# load model
model = load_model('model.h5')
X_t = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
X = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

res = model.predict(X)
print(res)