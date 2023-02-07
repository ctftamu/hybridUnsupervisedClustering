import numpy as np
import keras
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split

# Normalization constants
output_var_means = np.array([975.7077975425365, 0.4853567240255449, 5.177938468960296e-05, 3.8750383720470004e-05])
output_var_stdvs = np.array([523.4534762244654, 0.3299426636379226, 2.1084232618514112e-05, 1.5540012528054627e-05])
output_massfrac_means = np.array([0.017287042172356448, 0.0006306693804708345])
output_massfrac_stdvs = np.array([0.014417473283323897, 0.0010114137455385726])

""" Read Data """
data_file = 'data/SLM-Gaussian-4-vars.csv'
data = open(data_file)
n_clusters = int(data.readline())

input_features = [[] for i in range(n_clusters)]
outputs = [[] for i in range(n_clusters)]
clusters = [[] for i in range(n_clusters)]

for line in data:
    linedata = line.split(",")
    Zmean = float(linedata[1])
    Zvar = float(linedata[2])
    Chi = float(linedata[3])
    Temp = float(linedata[4])
    rho = float(linedata[5])
    diff = float(linedata[6])
    visc = float(linedata[7])

    cluster_num = int(linedata[0])-1
    temp_cluster = np.zeros((n_clusters,), dtype=int)
    temp_cluster[cluster_num] = 1

    input_features[cluster_num].append([Zmean, Zvar, Chi])
    outputs[cluster_num].append([Temp, rho, diff, visc])
    clusters[cluster_num].append(temp_cluster)

data.close()

# 80-20 train test split
train_features, test_features, train_targets, test_targets, train_clusters, test_clusters = train_test_split(input_features, outputs, clusters, test_size=0.2, shuffle=True)

""" Train Expert Models """
expert_models = []
for expert_index in range(n_clusters):
    # Build model architecture
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(train_features[0].shape[1],)))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(test_targets[0].shape[1]))

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    # Train model
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.5, patience=10, min_lr=0.0001, verbose=0, min_delta=0.005)  # Adaptive learning rate
    history = model.fit(train_features[expert_index], train_targets[expert_index],
                        validation_data=(test_features[expert_index], test_targets[expert_index]),
                        epochs=100, batch_size=256, callbacks=[reduce_lr])
    model_name = "SLM-Gaussian-"+str(n_clusters)+"-C"+str(expert_index+1)+".h5"
    model.save(model_name)
    expert_models.append(model)

# Concatenate data
all_train_features = train_features.reshape(-1, len(train_features[0][0]))
all_test_features = test_features.reshape(-1, len(test_features[0][0]))
all_train_targets = train_targets.reshape(-1, len(train_targets[0][0]))
all_test_targets = test_targets.reshape(-1, len(test_targets[0][0]))
all_train_clusters = train_clusters.reshape(-1, len(train_clusters[0][0]))
all_test_clusters = test_clusters.reshape(-1, len(test_clusters[0][0]))

""" Train Gating Network """
# Build model architecture
gating_net = keras.models.Sequential()
gating_net.add(layers.Dense(8, activation='relu', input_shape=(train_features[0].shape[1],)))
gating_net.add(layers.Dense(32, activation='relu'))
gating_net.add(layers.Dropout(0.025))
gating_net.add(layers.Dense(64, activation='relu'))
gating_net.add(layers.Dropout(0.025))
gating_net.add(layers.Dense(32, activation='relu'))
gating_net.add(layers.Dropout(0.025))
gating_net.add(layers.Dense(16, activation='relu'))
gating_net.add(layers.Dense(n_clusters, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
gating_net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
gating_net.fit(all_train_features, all_train_clusters, validation_data=(all_test_features, all_test_clusters), epochs=200, batch_size=256, verbose=0)
gating_net.save("SLM-Gaussian-"+str(n_clusters)+"-gate.h5")

""" Testing Models """
# Predict cluster using gating network
gating_preds = gating_net.predict(all_test_features)
pred_clusters = np.argmax(gating_preds, axis=1)
true_clusters = np.argmax(all_test_clusters, axis=1)

# Use cluster predictions to group inputs for expert prediction
cluster_features = [[] for i in range(n_clusters)]
cluster_targets = [[] for i in range(n_clusters)]
cluster_clusters = [[] for i in range(n_clusters)]
for i in range(len(all_test_features)):
    cluster_features[pred_clusters[i]].append(all_test_features[i])
    cluster_targets[pred_clusters[i]].append(all_test_targets[i])
    cluster_clusters[pred_clusters[i]].append(true_clusters[i])

# Each expert model makes predictions
all_preds = []
for i in range(n_clusters):
    all_preds.append(expert_models[i].predict(np.array([cluster_features[i]]))[0])

# Get prediction error statistics
abs_error = []
errors = []
for i in range(n_clusters):
    for j in range(len(all_preds[i])):
        abs_error.append(abs(all_preds[i][j] - cluster_targets[i][j]))
        errors.append(all_preds[i][j] - cluster_targets[i][j])

abs_error = np.array(abs_error)
errors = np.array(errors)
abs_error *= output_var_stdvs
errors *= output_var_stdvs

print("--- Prediction Statistics ---")
print("MAE:", abs_error.mean(axis=0))
print("STD:", abs_error.std(axis=0))
print()
print("MAX:", abs_error.max(axis=0))
print("MIN:", abs_error.min(axis=0))

