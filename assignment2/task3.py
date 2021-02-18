import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import time
import numpy as np


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10] # ENDRE DENNE I 4A OG 4B
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    
    t0 = time.time()

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    t1 = time.time()
    time_orig = t1-t0
    print("Time Original: ", time_orig)

    use_improved_weight_init = True

    t2 = time.time()

    model_improved_weight = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
        
    trainer_improved_weight = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weight, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_improved_weight, val_history_improved_weight = trainer_improved_weight.train(
        num_epochs)

    t3 = time.time()
    time_weight = t3-t2

    print("Time Improved weight: ", time_weight)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model - Without improved weights", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_weight["loss"], f"Task 3a Model - With improved weights.", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], f"Task 2 Model - Without improved weights T = {round(time_orig,2)} sec")
    utils.plot_loss(
        val_history_improved_weight["accuracy"], f"Task 3a Model - With improved weights. T = {round(time_weight,2)} sec")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Number of Training Steps")

    plt.legend()
    plt.savefig("task3a_train_loss.png")
    plt.show()


    use_improved_sigmoid = True

    t4 = time.time()

    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
        
    trainer_improved_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_improved_sigmoid.train(
        num_epochs)

    t5 = time.time()
    time_sigmoid = t5-t4

    print("Time Improved sigmoid: ", time_sigmoid)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(
        train_history_improved_weight["loss"], "Task 3a Model - Without improved sigmoid", npoints_to_average=10)
    utils.plot_loss(train_history_improved_sigmoid["loss"],
                    "Task 3b Model - With improved sigmoid", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])

    utils.plot_loss(
        val_history_improved_weight["accuracy"], f"Task 3a Model - Without improved sigmoid. T = {round(time_weight,2)} sec")

    utils.plot_loss(val_history_improved_sigmoid["accuracy"], f"Task 3b Model - With improved sigmoid. T = {round(time_sigmoid,2)} sec")
    
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Number of Training Steps")

    plt.legend()
    plt.savefig("task3b_train_loss.png")
    plt.show()

    use_momentum = True
    learning_rate = 0.02

    t6 = time.time()

    model_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
        
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_momentum, val_history_momentum = trainer_momentum.train(
        num_epochs)

    t7 = time.time()
    time_momentum = t7-t6

    print("Time Momentum: ", time_momentum)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)

    utils.plot_loss(train_history_improved_sigmoid["loss"],
                    "Task 3b Model - Without momentum", npoints_to_average=10)

    utils.plot_loss(
        train_history_momentum["loss"], "Task 3c Model - With Momentum", npoints_to_average=10)
    

    plt.ylim([0, .4])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])

    utils.plot_loss(val_history_improved_sigmoid["accuracy"], f"Task 3b Model - Without Momentum. T = {round(time_sigmoid,2)} sec")

    utils.plot_loss(
        val_history_momentum["accuracy"], f"Task 3c Model - With Momentum. T = {round(time_momentum,2)} sec")
    
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3c_train_loss.png")
    plt.show()
