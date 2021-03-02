import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
from task2 import SoftmaxTrainer
import time

if __name__ == "__main__":

    run4ab = False
    run4de = True

    run4e = False

    num_epochs = 50
    learning_rate = .1
    batch_size = 32

    neurons_per_layer = [64, 10]
    neurons_per_layer_a = [64, 10]
    neurons_per_layer_b = [60, 60, 10]
    neurons_per_layer_c = [64] * 10 + [10]


    momentum_gamma = .9  # Task 3 hyperparameter
    learning_rate = 0.02
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    if run4ab:
        use_momentum = True
        learning_rate = 0.02
        use_improved_weight_init = True
        use_improved_sigmoid = True

        t6 = time.time()

        model_64 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer_64 = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_64, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history_64, val_history_64 = trainer_64.train(
            num_epochs)

        t7 = time.time()
        time_64 = t7-t6

        print("Time 64 hidden units: ", time_64)

        neurons_per_layer = [32, 10]

        t8 = time.time()

        model_32 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer_32 = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_32, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history_32, val_history_32 = trainer_32.train(
            num_epochs)

        t9 = time.time()
        time_32 = t9-t8

        print("Time 32 hidden units: ", time_32)

        neurons_per_layer = [128, 10]

        t10 = time.time()

        model_128 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer_128 = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_128, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history_128, val_history_128 = trainer_128.train(
            num_epochs)

        t11 = time.time()
        time_128 = t11-t10

        print("Time 128 hidden units: ", time_128)

        plt.figure(figsize=(20, 12))


        plt.subplot(1, 1, 1)
        plt.ylim([0.85, 1])

        utils.plot_loss(val_history_32["accuracy"], f"Hidden units = 32. T = {round(time_32,2)} sec")

        utils.plot_loss(val_history_64["accuracy"], f"Hidden units = 64. T = {round(time_64,2)} sec")
        
        utils.plot_loss(val_history_128["accuracy"], f"Hidden units = 128. T = {round(time_128,2)} sec")

        plt.xlabel("Number of Training Steps")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.savefig("task4ab_accuracy.png")
        plt.show()




    if run4de:

        t0 = time.time()

        model_a = SoftmaxModel(
            neurons_per_layer_a,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer_a = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_a, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history_a, val_history_a = trainer_a.train(
            num_epochs)

        t1 = time.time()
        time_a = t1-t0

        print("Time a: ", time_a)

        t2 = time.time()

        model_b = SoftmaxModel(
            neurons_per_layer_b,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer_b = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_b, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history_b, val_history_b = trainer_b.train(
            num_epochs)

        t3 = time.time()
        time_b = t3-t2

        print("Time b: ", time_b)

        if run4e:
        
            t4 = time.time()

            model_c = SoftmaxModel(
                neurons_per_layer_c,
                use_improved_sigmoid,
                use_improved_weight_init)
                
            trainer_c = SoftmaxTrainer(
                momentum_gamma, use_momentum,
                model_c, learning_rate, batch_size, shuffle_data,
                X_train, Y_train, X_val, Y_val,
            )

            train_history_c, val_history_c = trainer_c.train(
                num_epochs)

            t5 = time.time()
            time_c = t5-t4

            print("Time c: ", time_c)
        

        ### PLOTTING

        plt.figure(figsize=(20,12))

        
        plt.subplot(1, 2, 1)

        utils.plot_loss(train_history_a["loss"],
                        "Training Loss - 1 layer", npoints_to_average=10)

        utils.plot_loss(train_history_b["loss"],
                        "Training Loss - 2 layers", npoints_to_average=10)
        if run4e:
            utils.plot_loss(train_history_c["loss"],
                            "Training Loss - 10 layers", npoints_to_average=10)


        utils.plot_loss(val_history_a["loss"],
                        "Validation Loss - 1 layer", npoints_to_average=10)

        utils.plot_loss(val_history_b["loss"],
                        "Validation Loss - 2 layers", npoints_to_average=10)
        if run4e:
            utils.plot_loss(val_history_c["loss"],
                            "Validation Loss - 10 layers", npoints_to_average=10)


        plt.ylim([0, .6])
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Cross Entropy Loss - Average")
        plt.legend()

            

        plt.subplot(1, 2, 2)
        plt.ylim([0.85, 1.02])

        utils.plot_loss(train_history_a["accuracy"], f"Training accuracy - 1 layer. T = {round(time_a,2)} sec")

        utils.plot_loss(train_history_b["accuracy"], f"Training accuracy - 2 layers. T = {round(time_b,2)} sec")

        if run4e:
            utils.plot_loss(train_history_c["accuracy"], f"Training accuracy - 10 layers. T = {round(time_c,2)} sec")


        utils.plot_loss(val_history_a["accuracy"], "Validation accuracy - 1 layer")

        utils.plot_loss(val_history_b["accuracy"], "Validation accuracy - 2 layers")
        
        if run4e:
            utils.plot_loss(val_history_c["accuracy"], "Validation accuracy - 10 layers")


        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.legend()
        if run4e:
            plt.savefig("task4de.png")
        else:
            plt.savefig("task4d.png")
        plt.show()
