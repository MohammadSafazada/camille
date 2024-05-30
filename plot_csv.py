import pandas as pd
import matplotlib.pyplot as plt

def plotting_function():
    # Load the CSV file
    file_path = 'loss_history.csv'  # Replace with your file path
    data = pd.read_csv(file_path)

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(data['Epoch'], data['Loss'], marker='o', linestyle='-', color='b', label='Training Loss')

    # Check if validation loss is present in the CSV and plot if it exists
    if 'Validation Loss' in data.columns:
        plt.plot(data['Epoch'], data['Validation Loss'], marker='o', linestyle='-', color='r', label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()