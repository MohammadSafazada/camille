81 % de l'espace de stockage utilisés … Une fois la limite atteinte, vous ne pouvez plus créer, modifier ni importer de fichiers. Profitez de 100 Go de stockage pour 2.00 CHF 0.50 CHF pendant 1 mois.
import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, train_and_evaluate
import time
from plot_csv import plotting_function
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)

    if args.nn_type == 'mlp':
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
    else : 
        xtrain = xtrain.reshape((xtrain.shape[0], 1, int(np.sqrt(xtrain.shape[1])), int(np.sqrt(xtrain.shape[1]))))
        xtest = xtest.reshape((xtest.shape[0], 1, int(np.sqrt(xtest.shape[1])), int(np.sqrt(xtest.shape[1]))))
    
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    print("shape of ytrain", ytrain.shape)
    # Make a validation set # overwrite + create ytest
    if not args.test:
    ### WRITE YOUR CODE HERE
        fraction_train = 0.8
        rinds = np.random.permutation(xtrain.shape[0]) # shuffling of the indices to shuffle the data

        n_train = int(xtrain.shape[0] * fraction_train)

        xtest = xtrain[rinds[n_train:]]
        xtrain = xtrain[rinds[:n_train]] 

        ytest = ytrain[rinds[n_train:]] 
        ytrain = ytrain[rinds[:n_train]]  

        print("shape of ytrain after validation set", ytrain.shape)

    ### WRITE YOUR CODE HERE to do any other data processing
    mean_val = np.mean(xtrain, axis=0, keepdims=True)
    std_val  = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, mean_val, std_val)
    xtest = normalize_fn(xtest, mean_val, std_val)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        
        pca_obj = PCA(args.pca_d)

        exvar = pca_obj.find_principal_components(xtrain)
        print(f'the expected variance with d = {args.pca_d} is: exvar = {exvar:.3f} %')
    
        pca_obj.reduce_dimension(xtest)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes, num_hidden_layers=args.hd, max_neurons=args.max_neur) 
    elif args.nn_type == "cnn":
        model = CNN(input_channels = 1,n_classes =10) 
    elif args.nn_type == "transformer":
        model = MyViT(chw=(1, 28, 28), n_patches=7, n_blocks=4, hidden_d=256, n_heads=8, out_d=n_classes) 

    summary(model)

    # Trainer object
    print("model is ", args.nn_type)
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    ## 4. Train and evaluate the method
    ## 4. Train and evaluate the method
     ### finding best hyperparameters ###
    if args.find_BH == "yes":
        train_dataset = TensorDataset(torch.tensor(xtrain, dtype=torch.float32), torch.tensor(ytrain, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(xtest, dtype=torch.float32), torch.tensor(ytest, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        lrs = [ 0.003, 0.004, 0.005, 0.006,]
        epochs_list = [5, 10]

        for lr in lrs:
            for epochs in epochs_list:
                model = CNN(input_channels=1, n_classes=10)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr)

                print(f"Training with lr={lr} and epochs={epochs}")
                train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs)


    # Fit (:=train) the method on the training data
    t1 = time.time()
    if not args.test: 
        # calculate the loss of the train and the validation set to compare 
        preds_train = method_obj.fit(xtrain, ytrain, xtest, ytest) 
    else : 
        preds_train = method_obj.fit(xtrain, ytrain)
    # Predict on unseen data
    preds = method_obj.predict(xtest)
    t2 = time.time()
    ## Report results: performance on train and valid/test sets
    print(f"\nMethod {args.nn_type} takes {(t2-t1):.5f} seconds")
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    if not args.test : #accuracy with ytest
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        if args.plot_loss : 
            plotting_function() 
        
    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    # if not args.test:
    #     if args.plot == 'find_best_lr':
    #         plot_acc(model, xtrain, xtest, ytrain, ytest, args.nn_batch_size, args.max_iters)
    #     #if args.plot == 'expvar':
    #         #plot_expvar()


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=10, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    parser.add_argument('--plot_loss', action="store_true", help="plot loss functions")
    parser.add_argument('--plot', default="no", type=str, help="it can be 'find_best_lr' | 'expvar' | '' ")
    parser.add_argument('--find_BH', default="no", type=str, help="active search for best hyper parameter if 'yes'")
    parser.add_argument('--hd', type=int, default=3, help="Number of hidden layers")
    parser.add_argument('--max_neur', type=int, default=512)
    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)