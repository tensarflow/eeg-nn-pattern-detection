from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
import numpy as np


dataset = read_csv('eegdata.csv')
# manually specify column names
dataset.columns = ['ch_1', 'ch_2', 'ch_3', 'ch_4', 'ch_5', 'ch_6', 'ch_7', 'ch_8', 'ch_9', 'ch_10', 'ch_11', 'ch_12', 'ch_13', 'ch_14', 'status']

# collect train dataset for left side
train_data_left = []
train_data_left.append(dataset[140 : 140+360])
train_data_left.append(dataset[1270 : 1270+360])
train_data_left.append(dataset[2116 : 2116+360])
train_data_left.append(dataset[3274 : 3274+360])
train_data_left.append(dataset[5184 : 5184+360])
train_data_left.append(dataset[6600 : 6600+360])
train_data_left.append(dataset[11046 : 11046+360])

# collect train dataset for right side
train_data_right = []
train_data_right.append(dataset[800 : 800+360])
train_data_right.append(dataset[1560 : 1560+360])
train_data_right.append(dataset[2558 : 2558+360])
train_data_right.append(dataset[4272 : 4272+360])
train_data_right.append(dataset[5849 : 5184+360])
train_data_right.append(dataset[8983 : 6600+360])
train_data_right.append(dataset[12161 : 12161+360])

print(len(train_data_left))
# plot dataset

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# plot each column
i = 1
pyplot.figure()
for seq in range(len(train_data_left)):
	values = train_data_left[seq].values
	for group in groups:
		pyplot.subplot(7, len(groups), i)
		pyplot.plot(values[:, group])
		pyplot.title(train_data_left[seq].columns[group], y=0.5, loc='right')
		i += 1
pyplot.show()



# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = train_images / 255.0
# test_images = test_images / 255.0

# model = keras.Sequential([
	# keras.layers.Flatten(input_shape=(28,28)),
	# keras.layers.Dense(128, activation=tf.nn.relu),
	# keras.layers.Dense(128, activation=tf.nn.relu),
	# keras.layers.Dense(128, activation=tf.nn.relu),
	# keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer=tf.train.AdamOptimizer(),
	# loss='sparse_categorical_crossentropy',
	# metrics=['accuracy'])
	
# model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy', test_acc)

# predictions = model.predict(test_images)
# print(predictions[0])
# print(np.argmax(predictions[0]))
# print(test_labels[0])

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
  # plt.subplot(num_rows, 2*num_cols, 2*i+1)
  # plot_image(i, predictions, test_labels, test_images)
  # plt.subplot(num_rows, 2*num_cols, 2*i+2)
  # plot_value_array(i, predictions, test_labels)
# plt.show()