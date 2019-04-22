from keras.models import load_model
import numpy as np

encoder = load_model(r'./weights/encoder_weights0.h5')
decoder = load_model(r'./weights/decoder_weights0.h5')

inputs = np.array([[0.61538464,0.79963964,0.24822696,0.0356072,0.02318621,0.19659062,0.10462296,0.04426264,0.125019]])
x = encoder.predict(inputs)
y = decoder.predict(x)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(x))
print('Decoded: {}'.format(y))