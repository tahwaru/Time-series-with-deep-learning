# A temperature forecasting problem
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

#--------------------investigating the content of the csv file----
data_dir = '/home/tahawaru/Aiml/TimesSeries_Weather'
file_name = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(file_name)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
#print(header)
#print(len(lines))


#---------
num_lines_of_data = len(lines)
float_data = np.zeros((num_lines_of_data, len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values
    
temp = float_data[:,1]
plt.figure(1)
plt.plot(range(len(temp)), temp)
#plt.show()
plt.figure(2)
plt.plot(range(len(temp) // 1000), temp[:len(temp) // 1000])
#plt.show()
# Fig 2 shows temperature below 0 degree. this means it
plt.figure(3)
plt.plot(range(60 * 24 // 10), temp[:60*24 // 10])
plt.title("1-day temperature") # 1-day --> 144 points of data given 1 recording each 10 minutes
#plt.show()

#--------Preparing the data for timeseries prediction
lookback = 720 # Observations will go back to 5 days (5 * 144 )
step = 6 # Observations will be sampled at one data point per hour
delay = 144 # Targets will be 24 hours in the future (how many timesteps in the future the target should be)
batch_size = 128
#------Normalizing the data (subtracting the mean and dividing by the std for the training data only)
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

#-----generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets =  np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets



#-------------Using the generator to instantiate the training, validation and test sets
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=200001,
                      max_index=300000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
test_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=300001,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(float_data) - 300001 - lookback) # how many steps to draw from test_gen in order to see the entire test set

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
#evaluate_naive_method()


#--------------Small densely connected model
input_tensor = tf.keras.layers.Input(shape=(lookback // step, float_data.shape[-1]))
x = tf.keras.layers.Flatten()(input_tensor)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output_tensor = tf.keras.layers.Dense(1)(x)

model=tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_gen,
                              epochs=20,
                              validation_data=val_gen,
                              callbacks=callbacks)

#-----plotting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
