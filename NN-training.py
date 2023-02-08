import numpy as np
import keras
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from typing import List


def read_data(data_file: str, n_inputs: int = 3, n_outputs: int = 4, test_split: float = 0.2):
    """
    Reads data from csv holding cluster information. The data should
    use the following format.
        "cluster, Zmean, Zvar, Chi, Temp, rho, diff, visc"
        "cluster, Zmean, Zvar, Chi, CO, OH"
    The first row holds the number of clusters.

    Args:
        data_file: string with data file path
        n_inputs: int of number of inputs (3 for Zmean, Zvar, Chi)
        n_outputs: int of number of outputs (4 for Temp, rho, diff, visc)
                                         or (2 for CO, OH)
        test_split: float representing portion of data to use for testing

    Returns:
        train/test_features: Numpy array of normalized features
        train/test_targets: Numpy array of normalized targets
        train/test_clusters: Numpy array of classification labels
    """
    # Read Data
    data = open(data_file)
    n_clusters = int(data.readline())

    input_features = [[] for i in range(n_clusters)]
    output_targets = [[] for i in range(n_clusters)]
    clusters = [[] for i in range(n_clusters)]

    for line in data:
        linedata = line.split(",")
        inputs = []
        for i in range(n_inputs):
            inputs.append(float(linedata[i+1]))

        outputs = []
        for i in range(n_outputs):
            outputs.append(float(linedata[i+1+n_inputs]))

        cluster_num = int(linedata[0])-1
        temp_cluster = np.zeros((n_clusters,), dtype=int)  # One-hot encoding
        temp_cluster[cluster_num] = 1

        input_features[cluster_num].append(inputs)
        output_targets[cluster_num].append(outputs)
        clusters[cluster_num].append(temp_cluster)

    data.close()

    # Split data for training and testing
    train_features, test_features, train_targets, test_targets, train_clusters, test_clusters = train_test_split(input_features, output_targets, clusters, test_size=test_split, shuffle=True)
    return train_features, test_features, train_targets, test_targets, train_clusters, test_clusters


def train_experts(train_features, test_features, train_targets, test_targets, save_models=True, verbose=False):
    """
    Trains expert models using clustered data.

    Args:
        train_features: List of normalized training features (n_clusters)
        test_features: List of normalized testing features (n_clusters)
        train_targets: List of normalized training targets (n_clusters)
        test_targets: List of normalized testing targets (n_clusters)
        save_models: bool to determine if models will be saved
        verbose: bool to determine if model training is verbose

    Returns:
        expert_models: List of Keras models (n_clusters)
    """
    n_clusters = len(train_features)

    # Train Expert Models
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
        model.add(layers.Dense(train_targets[0].shape[1]))

        # Compile model
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

        # Train model
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.5, patience=10, min_lr=0.0001, verbose=verbose, min_delta=0.005)  # Adaptive learning rate
        history = model.fit(train_features[expert_index], train_targets[expert_index],
                            validation_data=(test_features[expert_index], test_targets[expert_index]),
                            epochs=100, batch_size=256, callbacks=[reduce_lr])
        if save_models:
            model_name = "SLM-Gaussian-"+str(n_clusters)+"-C"+str(expert_index+1)+".h5"
            model.save(model_name)
        expert_models.append(model)

    return expert_models


def combine_cluster_data(train_features, test_features, train_targets, test_targets, train_clusters, test_clusters):
    """
    Combines separated cluster data into a single list.

    Args:
        train_features: List of normalized training features (n_clusters)
        test_features: List of normalized testing features (n_clusters)
        train_targets: List of normalized training targets (n_clusters)
        test_targets: List of normalized testing targets (n_clusters)
        train_clusters: List of one-hot encoded cluster training labels (n_clusters)
        test_clusters: List of one-hot encoded cluster testing labels (n_clusters)

    Returns:
        combined_train_features: List of normalized training features
        combined_test_features: List of normalized testing features
        combined_train_targets: List of normalized training targets
        combined_test_targets: List of normalized testing targets
        combined_train_clusters: List of one-hot encoded cluster training labels
        combined_test_clusters: List of one-hot encoded cluster testing labels
    """
    # Concatenate data
    combined_train_features = train_features.reshape(-1, len(train_features[0][0]))
    combined_test_features = test_features.reshape(-1, len(test_features[0][0]))
    combined_train_targets = train_targets.reshape(-1, len(train_targets[0][0]))
    combined_test_targets = test_targets.reshape(-1, len(test_targets[0][0]))
    combined_train_clusters = train_clusters.reshape(-1, len(train_clusters[0][0]))
    combined_test_clusters = test_clusters.reshape(-1, len(test_clusters[0][0]))

    return combined_train_features, combined_test_features, combined_train_targets, combined_test_targets, combined_train_clusters, combined_test_clusters


def train_gating_net(combined_train_features, combined_test_features, combined_train_clusters, combined_test_clusters, save_model=True, verbose=False):
    """
    Trains gating network to predict clusters.

    Args:
        combined_train_features: List of normalized training features
        combined_test_features: List of normalized testing features
        combined_train_clusters: List of one-hot encoded cluster training labels
        combined_test_clusters: List of one-hot encoded cluster testing labels
        save_model: bool to determine if models will be saved
        verbose: bool to determine if model training is verbose

    Returns:
        gating_net: gating network model - Keras Model
    """
    n_clusters = len(combined_train_clusters[0])

    # Build model architecture
    gating_net = keras.models.Sequential()
    gating_net.add(layers.Dense(8, activation='relu', input_shape=(combined_train_features[0].shape[1],)))
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
    gating_net.fit(combined_train_features, combined_train_clusters,
                   validation_data=(combined_test_features, combined_test_clusters),
                   epochs=200, batch_size=256, verbose=verbose)
    if save_model:
        gating_net.save("SLM-Gaussian-"+str(n_clusters)+"-gate.h5")

    return gating_net


def evaluate_models(expert_models: List[keras.Model], gating_net: keras.Model, combined_test_features, combined_test_targets, combined_test_clusters, output_stdvs):
    """
    Evaluates trained expert models and gating network.

    Args:
        expert_models: List of Keras models (n_clusters)
        gating_net: gating network model - Keras Model
        combined_test_features: List of normalized testing features
        combined_test_targets: List of normalized testing targets
        combined_test_clusters: List of one-hot encoded cluster testing labels

    Returns:
        None
    """
    n_clusters = len(expert_models)

    # Predict cluster using gating network
    gating_preds = gating_net.predict(combined_test_features)
    pred_clusters = np.argmax(gating_preds, axis=1)
    true_clusters = np.argmax(combined_test_clusters, axis=1)

    # Use cluster predictions to group inputs for expert prediction
    cluster_features = [[] for i in range(n_clusters)]
    cluster_targets = [[] for i in range(n_clusters)]
    cluster_clusters = [[] for i in range(n_clusters)]
    for i in range(len(combined_test_features)):
        cluster_features[pred_clusters[i]].append(combined_test_features[i])
        cluster_targets[pred_clusters[i]].append(combined_test_targets[i])
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
    abs_error *= output_stdvs
    errors *= output_stdvs

    print("--- Prediction Statistics ---")
    print("MAE:", abs_error.mean(axis=0))
    print("STD:", abs_error.std(axis=0))
    print()
    print("MAX:", abs_error.max(axis=0))
    print("MIN:", abs_error.min(axis=0))

