import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os
from PIL import Image
# Function to prepare the MNIST dataset
def prepare_mnist_dataset():
    dataset = []
    data_directory = "."
    download_dataset = True
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(data_directory, train=True, download=download_dataset, transform=transform)
    mnist_test = datasets.MNIST(data_directory, train=False, download=download_dataset, transform=transform)
    mnist_train, mnist_validation = data.random_split(mnist_train, (48000, 12000))
    dataset.append(mnist_train)
    dataset.append(mnist_validation)
    dataset.append(mnist_test)
    return dataset
# Definition of the neural network architecture
class ArtificialNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.input_layer = nn.Linear(28*28, 64)  # Input layer
        self.act1 = nn.ReLU() #hidden layer 1 activation function
        self.hidden_layer2 = nn.Linear(64, 64)  # hidden layer 2 weights
        self.act2 = nn.ReLU() #hidden layer 2 activation function
        self.output_layer = nn.Linear(64, 10)  # Output layer        
    def forward(self, features): 
        features = features.view(-1, 28*28)  # Flatten the input tensor
        features = self.act1(self.input_layer(features)) 
        features = self.dropout(features)
        features = self.act2(self.hidden_layer2(features)) 
        features = self.dropout(features)
        features = self.output_layer(features)
        return torch.softmax(features, dim=1)
# Function to calculate the accuracy
def calculate_accuracy(outputs, expected):
    pred = outputs.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()
# Function to train the neural network
def train_neural_network():   
    print("Training model...")
    model = ArtificialNeuralNetwork()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    train_loss = []
    validation_accuracy = []
    best_accuracy = -1
    best_epoch = -1
    max_epochs = 100
    no_improvement_epochs = 5
    batch_size = 512
    mnist_data = prepare_mnist_dataset()
    mnist_train = mnist_data[0]
    mnist_validation = mnist_data[1]
    mnist_test = mnist_data[2]
    for epoch in range(max_epochs):
        model.train()
        train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            model_predictions = model(X_batch)
            loss = loss_function(model_predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())  
        train_loss.append(torch.tensor(epoch_losses).mean())
        model.eval()
        validation_loader = data.DataLoader(mnist_validation, batch_size=batch_size, shuffle=False)
        X_valid, y_valid = next(iter(validation_loader))
        model_predictions = model(X_valid)
        accuracy = calculate_accuracy(model_predictions, y_valid).detach()
        validation_accuracy.append(accuracy)        
        if best_accuracy == -1 or accuracy > best_accuracy:
            print("New best epoch ", epoch, "accuracy", accuracy)
            best_accuracy = accuracy
            best_model = model.state_dict()
            best_epoch = epoch  
        if best_epoch + no_improvement_epochs <= epoch:
            print("Done!\n")
            break
    # Compute final accuracy on test set
    model.eval()
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    X_test, y_test = next(iter(test_loader))
    test_predictions = model(X_test)
    final_accuracy = calculate_accuracy(test_predictions, y_test).item()
    # Create log.txt file
    with open("log.txt", "w") as f:
        f.write("Epoch	    Train Loss	    Validation Accuracy\n")
        for epoch, (loss, acc) in enumerate(zip(train_loss, validation_accuracy)):
            f.write(f"{epoch}   	{loss}\t{acc}\n")
        f.write(f"\nFinal Accuracy on Test Set: {final_accuracy}\n")
    
    model.load_state_dict(best_model)
    return model
# Function to handle user input and make predictions
def handle_user_input(model):
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])  # Define the transformation
    while True:
        filepath = input("Please enter a filepath:\n> ")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break       
        if not os.path.isfile(filepath):
            print("File does not exist. Try again.")
            continue       
        try:
            with torch.no_grad():
                image = Image.open(filepath)
                image = transform(image)  # Apply the transformation
                prediction = model(image)
                predicted_label = prediction.argmax(dim=1).item()
                print(f"Classifier: {predicted_label}\n")
        except Exception as e:
            print(f"Error processing image: {e}")
# Main function to execute the program
def main():
    model = train_neural_network()  # Load or train the model as needed.
    handle_user_input(model)
if __name__ == "__main__":
    main()    