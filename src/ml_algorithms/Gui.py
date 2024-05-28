import tkinter as tk
from tkinter import messagebox
import random_forest
import knn
import decision_tree
import nn
import threading
import svm

descriptions = {
    "Random Forest": "Random Forest is an ensemble learning method for classification, regression, and other tasks that operates by constructing a multitude of decision trees at training time.",
    "KNN": "k-Nearest Neighbors (k-NN) is a non-parametric method used for classification and regression. In k-NN classification, the output is a class membership.",
    "Decision Tree": "Decision Tree is a tree-like model of decisions and their possible consequences. It is one way to display an algorithm that only contains conditional control statements.",
    "Neural Network": "Neural Networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or clustering of raw input.",
    "SVM": "Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification or regression tasks. It finds the hyperplane that best separates different classes in the feature space.",
}

# Create the main window
root = tk.Tk()
root.title("Algorithm Selector")
root.geometry("500x400")  # Set window size

# Welcome message
welcome_label = tk.Label(root, text="Welcome to our Genre-Detection App!\n"
                                    "The datasets are loaded in the algorithms.\n"
                                    "Please choose the algorithm and parameters you want,\n"
                                    "then run the algorithm to see the results",
                         wraplength=400, justify="center")
welcome_label.pack()
# Create a larger break between the opening text and the following text
tk.Label(root, text="").pack()
# Dropdown for selecting algorithm
algorithm_label = tk.Label(root, text="Select Algorithm:")
algorithm_label.pack()

algorithms = ["Random Forest", "KNN", "Decision Tree", "Neural Network", "SVM"]
algorithm_variable = tk.StringVar(root)
algorithm_variable.set(algorithms[0])
algorithm_dropdown = tk.OptionMenu(root, algorithm_variable, *algorithms)
algorithm_dropdown.pack()

# Description label
description_label = tk.Label(root, text="", wraplength=400, justify="left")
description_label.pack()

# Status label
status_label = tk.Label(root, text="", fg="blue")
status_label.pack()

# Number of Epochs entry for Neural Network
epochs_label = tk.Label(root, text="Number of Epochs:")
epochs_entry = tk.Entry(root)

# Learning Rate entry for Neural Network
learning_rate_label = tk.Label(root, text="Learning Rate:")
learning_rate_entry = tk.Entry(root)

# K entry for KNN
k_label = tk.Label(root, text="Enter K:")
k_entry = tk.Entry(root)

# Random Forest parameters
rf_parameters_label = tk.Label(root, text="Random Forest Parameters:")
n_estimators_label = tk.Label(root, text="n_estimators:")
n_estimators_entry = tk.Entry(root)
max_depth_label = tk.Label(root, text="max_depth:")
max_depth_entry = tk.Entry(root)
min_samples_split_label = tk.Label(root, text="min_samples_split:")
min_samples_split_entry = tk.Entry(root)
min_samples_leaf_label = tk.Label(root, text="min_samples_leaf:")
min_samples_leaf_entry = tk.Entry(root)
criterion_label = tk.Label(root, text="criterion:")
criterion_variable = tk.StringVar(root)
criterion_variable.set("gini")
criterion_dropdown = tk.OptionMenu(root, criterion_variable, "gini", "entropy")

# Decision Tree parameters
dt_parameters_label = tk.Label(root, text="Decision Tree Parameters:")
dt_max_depth_label = tk.Label(root, text="Max Depth:")
dt_max_depth_entry = tk.Entry(root)
dt_criterion_label = tk.Label(root, text="Criterion:")
dt_criterion_variable = tk.StringVar(root)
dt_criterion_variable.set("gini")
dt_criterion_dropdown = tk.OptionMenu(root, dt_criterion_variable, "gini", "entropy")
dt_min_samples_split_label = tk.Label(root, text="Min Samples Split:")
dt_min_samples_split_entry = tk.Entry(root)
dt_min_samples_leaf_label = tk.Label(root, text="Min Samples Leaf:")
dt_min_samples_leaf_entry = tk.Entry(root)

# SVM parameters
svm_parameters_label = tk.Label(root, text="SVM Parameters:")
svm_C_label = tk.Label(root, text="C:")
svm_C_entry = tk.Entry(root)
svm_kernel_label = tk.Label(root, text="Kernel:")
svm_kernel_variable = tk.StringVar(root)
svm_kernel_variable.set("linear")
svm_kernel_dropdown = tk.OptionMenu(root, svm_kernel_variable, "linear", "rbf", "poly", "sigmoid")

# Function to update description label
def update_description(*args):
    selected_algorithm = algorithm_variable.get()
    description_label.config(text=descriptions.get(selected_algorithm, ""))
    # Show additional inputs for Neural Network
    if selected_algorithm == "Neural Network":
        epochs_label.pack()
        epochs_entry.pack()
        learning_rate_label.pack()
        learning_rate_entry.pack()
    else:
        epochs_label.pack_forget()
        epochs_entry.pack_forget()
        learning_rate_label.pack_forget()
        learning_rate_entry.pack_forget()

    # Show additional input for KNN
    if selected_algorithm == "KNN":
        k_label.pack()
        k_entry.pack()
    else:
        k_label.pack_forget()
        k_entry.pack_forget()

    # Show additional inputs for Random Forest
    if selected_algorithm == "Random Forest":
        rf_parameters_label.pack()
        n_estimators_label.pack()
        n_estimators_entry.pack()
        max_depth_label.pack()
        max_depth_entry.pack()
        min_samples_split_label.pack()
        min_samples_split_entry.pack()
        min_samples_leaf_label.pack()
        min_samples_leaf_entry.pack()
        criterion_label.pack()
        criterion_dropdown.pack()
    else:
        rf_parameters_label.pack_forget()
        n_estimators_label.pack_forget()
        n_estimators_entry.pack_forget()
        max_depth_label.pack_forget()
        max_depth_entry.pack_forget()
        min_samples_split_label.pack_forget()
        min_samples_split_entry.pack_forget()
        min_samples_leaf_label.pack_forget()
        min_samples_leaf_entry.pack_forget()
        criterion_label.pack_forget()
        criterion_dropdown.pack_forget()

    # Show additional inputs for Decision Tree
    if selected_algorithm == "Decision Tree":
        dt_parameters_label.pack()
        dt_max_depth_label.pack()
        dt_max_depth_entry.pack()
        dt_min_samples_split_label.pack()
        dt_min_samples_split_entry.pack()
        dt_min_samples_leaf_label.pack()
        dt_min_samples_leaf_entry.pack()
        dt_criterion_label.pack()
        dt_criterion_dropdown.pack()
    else:
        dt_parameters_label.pack_forget()
        dt_max_depth_label.pack_forget()
        dt_max_depth_entry.pack_forget()
        dt_criterion_label.pack_forget()
        dt_criterion_dropdown.pack_forget()
        dt_min_samples_split_label.pack_forget()
        dt_min_samples_split_entry.pack_forget()
        dt_min_samples_leaf_label.pack_forget()
        dt_min_samples_leaf_entry.pack_forget()

    # Show additional inputs for SVM
    if selected_algorithm == "SVM":
        svm_parameters_label.pack()
        svm_C_label.pack()
        svm_C_entry.pack()
        svm_kernel_label.pack()
        svm_kernel_dropdown.pack()
    else:
        svm_parameters_label.pack_forget()
        svm_C_label.pack_forget()
        svm_C_entry.pack_forget()
        svm_kernel_label.pack_forget()
        svm_kernel_dropdown.pack_forget()

# Initial update
update_description()

result_label = tk.Label(root, text="")
result_label.pack(side=tk.BOTTOM, anchor=tk.S, pady=10)

# Button to run the selected algorithm
def run_algorithm():
    status_label.config(text="Running")
    selected_algorithm = algorithm_variable.get()
    if selected_algorithm == "Random Forest":
        n_estimators = int(n_estimators_entry.get())
        max_depth = int(max_depth_entry.get())
        min_samples_split = int(min_samples_split_entry.get())
        min_samples_leaf = int(min_samples_leaf_entry.get())
        criterion = criterion_variable.get()
        result = random_forest.main(n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf)
    elif selected_algorithm == "KNN":
        k_value = int(k_entry.get())
        result = knn.main(k_value)
    elif selected_algorithm == "Decision Tree":
        max_depth = int(dt_max_depth_entry.get())
        criterion = dt_criterion_variable.get()
        min_samples_split = int(dt_min_samples_split_entry.get())
        min_samples_leaf = int(dt_min_samples_leaf_entry.get())
        result = decision_tree.main(max_depth, criterion, min_samples_split, min_samples_leaf)
    elif selected_algorithm == "Neural Network":
        epochs = int(epochs_entry.get())
        learning_rate = float(learning_rate_entry.get())
        result = nn.main(epochs, learning_rate)
    elif selected_algorithm == "SVM":
        C = float(svm_C_entry.get())
        kernel = svm_kernel_variable.get()
        result = svm.main(C, kernel)
    else:
        messagebox.showerror("Error", "Please select an algorithm")
        return
    
    result_label.config(text=result)
    # Update status label
    status_label.config(text="Completed")

run_button = tk.Button(root, text="Run Algorithm", command=run_algorithm)
run_button.pack()

# Bind dropdown selection to update description
algorithm_variable.trace_variable("w", lambda *args: update_description())

# Run the main event loop
root.mainloop()