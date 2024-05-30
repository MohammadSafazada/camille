import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
from src.utils import accuracy_fn
from torch.optim import Adam
import csv
## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, num_hidden_layers, max_neurons):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        in_size = input_size
        self.hidden_layers = num_hidden_layers
        
        for i in range(num_hidden_layers):
            neurons = max_neurons // (2 ** i)
            setattr(self, f'fc{i+1}', nn.Linear(in_size, neurons))
            in_size = neurons
        
        self.fc_out = nn.Linear(in_size, n_classes)
        

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        # Input was already flatten in the main

        # apply the activation functions on the input:

        
        for hl in range(0, self.hidden_layers):
            x = F.relu(getattr(self, f'fc{hl+1}')(x))
        preds = self.fc_out(x)

        return preds

class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """

        super(CNN, self).__init__()

    
        ### MY code 2 ###
        self.conv1 = nn.Conv2d(input_channels, 6, 3, padding=1)  # (input_channels, 28, 28) -> (6, 28, 28)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)  # (6, 14, 14) -> (16, 14, 14)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # (16 * 7 * 7) -> 120
        self.fc2 = nn.Linear(120, n_classes)  # 120 -> n_classes
        ###

       

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
    
        ### MY CODE 2 ###
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (6, 28, 28) -> (6, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (16, 14, 14) -> (16, 7, 7)
        
        # Flatten the images into vectors
        x = x.view(-1, 16 * 7 * 7)  # Reshape to (N, 16 * 7 * 7)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        preds = self.fc2(x)
        ###

        return preds
      
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq) ### WRITE YOUR CODE HERE

                attention = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d) ### WRITE YOUR CODE HERE
        self.mhsa = MyMSA(hidden_d, n_heads) ### WRITE YOUR CODE HERE
        self.norm2 = nn.LayerNorm(hidden_d) ### WRITE YOUR CODE HERE
        self.mlp = nn.Sequential( ### WRITE YOUR CODE HERE
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out
    
class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        
        """
        Initialize the network.
        
        """
        super().__init__()
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) ### WRITE YOUR CODE HERE

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d) ### WRITE YOUR CODE HERE

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification linear because of the loss function (which needs data with not softmax applied)
        self.mlp = nn.Linear(self.hidden_d, out_d)
        
        
    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if i % 2 == 0:
                    result[i, j] = math.sin(i / (10000 ** (j / d)))
                else:
                    result[i, j] = math.cos(i / (10000 ** ((j - 1) / d)))
        return result

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #print("beginning forward and shape of x is ", x.shape)
        n, c, h, w = x.shape
        #print("before patches")
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches) ### WRITE YOUR CODE HERE
        #print("before tokens")
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE
        #print("before adding classification")
        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out) ### WRITE YOUR CODE HERE
        
        return out
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches ### WRITE YOUR CODE HERE

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):

                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] ### WRITE YOUR CODE HERE

                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten() ### WRITE YOUR CODE HERE

        return patches


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()

        if isinstance(model, CNN):
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif isinstance(model, MyViT):  # Remplacez par le nom de votre classe Transformer
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif isinstance(model, MLP):  # Remplacez par le nom de votre classe MLP
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError("Unknown model type")
        
        self.loss_history =[]
        self.loss_validation = []

    def train_all(self, dataloader,  val_dataloader=None):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            backward = True
            epoch_loss = self.train_one_epoch(dataloader, ep, backward)
            self.loss_history.append(epoch_loss)  # Store the epoch loss
            if val_dataloader is not None:
                self.backward = False
                epoch_loss_val = self.train_one_epoch(val_dataloader, ep, backward)
                self.loss_validation.append(epoch_loss_val)
        

        # Save the loss history to a CSV file at the end of training
        with open('loss_history.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if val_dataloader is not None:
                writer.writerow(['Epoch', 'Loss', 'Validation Loss'])
                for i in range(len(self.loss_history)):
                    writer.writerow([i + 1, self.loss_history[i], self.loss_validation[i]])
            else:
                writer.writerow(['Epoch', 'Loss'])
                for i in range(len(self.loss_history)):
                    writer.writerow([i + 1, self.loss_history[i]])

        

    def train_one_epoch(self, dataloader, ep, backward=True):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!
        
        Arguments:
            dataloader (DataLoader): dataloader for training data
            ep (int): current epoch number
            backward (bool): whether to perform the backward pass and update weights
        """
        if backward:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        for it, batch in enumerate(dataloader):
            x, y = batch
            #print("train_one_epoch")
            
            # Convert y to long if necessary
            if y.dtype != torch.long:
                y = y.long()

            # Run forward pass
            logits = self.model(x)

            # Compute loss
            loss = self.criterion(logits, y)

            if backward:
                # Run backward pass
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Zero-out the accumulated gradients
                self.optimizer.zero_grad()

            epoch_loss += loss.detach().cpu().item() / len(dataloader)
            print(f'\rEp {ep + 1}/{self.epochs}, it {it + 1}/{len(dataloader)}: loss train: {loss:.2f}', end='')

        return epoch_loss


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x = batch[0] if isinstance(batch, (tuple, list)) else batch

                # Run forward pass.
                logits = self.model(x)

                # Get the predicted labels.
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds)

        # Concatenate all predictions into a single tensor.
        pred_labels = torch.cat(all_preds)
        return pred_labels
    
    def fit(self, training_data, training_labels, validation_data=None, validation_labels=None):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # validation data 
        if validation_data is not None :  
            validation_dataset = TensorDataset(torch.from_numpy(validation_data).float(), 
                                      torch.from_numpy(validation_labels))
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)
        

        self.train_all(train_dataloader, validation_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()