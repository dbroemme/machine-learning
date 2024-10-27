import streamlit as st
import numpy as np
import plotly.graph_objects as go
from neuron import Neuron
from neural_network import SimpleNeuralNetwork
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Interactive Neuron Visualization", layout="wide")

st.title("Interactive Neural Network Visualization")


def create_3d_graph(X, Y, Z, title, uirevision_params):
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title=title,
        autosize=False,
        width=600,
        height=500,
        scene=dict(
            xaxis_title='Input 1',
            yaxis_title='Input 2',
            zaxis_title='Output'
        ),
        uirevision=str(uirevision_params)
    )
    return fig

# Create a function to draw the network
def draw_neural_network(nn):
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    node_sizes = []
    edge_labels = {}

    # Input layer
    G.add_node("I1")
    pos["I1"] = (0, 0)
    node_colors.append('lightblue')
    node_sizes.append(1000)

    # Hidden layer
    for i in range(3):
        node = f"H{i+1}"
        G.add_node(node)
        pos[node] = (1, i - 1)
        node_colors.append('lightgreen')
        node_sizes.append(1000)
        weight = nn.hidden_layer[i].weights[0]
        G.add_edge("I1", node)
        edge_labels[("I1", node)] = f"{weight:.2f}"

    # Output layer
    G.add_node("O1")
    pos["O1"] = (2, 0)
    node_colors.append('salmon')
    node_sizes.append(1000)

    for i in range(3):
        weight = nn.output_neuron.weights[i]
        G.add_edge(f"H{i+1}", "O1")
        edge_labels[(f"H{i+1}", "O1")] = f"{weight:.2f}"

    # Create the plot
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, 
            font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    # Add bias annotations
    for i, node in enumerate(["I1", "H1", "H2", "H3", "O1"]):
        if node == "I1":
            bias = 0
        elif node == "O1":
            bias = nn.output_neuron.bias
        else:
            bias = nn.hidden_layer[i-1].bias
        plt.annotate(f"b: {bias:.2f}", xy=pos[node], xytext=(15, 5), 
                     textcoords="offset points", fontsize=12)

    plt.title("Neural Network Structure with Weights and Biases")
    plt.axis('off')
    return plt

# Add this new section for the data input table
st.header("Training Data")

# Create a default dataframe
default_data = pd.DataFrame({
    'Input': [-1, 0, 1, 2, 3, 4],
    'Output': [4, 3, 2, 1, 0, -1]
})

# Create an editable data table
edited_df = st.data_editor(default_data, num_rows="dynamic")

# Add a button to train the network
if st.button("Train Neural Network"):
    # Extract input and output data from the dataframe
    X = edited_df['Input'].values.reshape(-1, 1)
    y = edited_df['Output'].values

    # Create a SimpleNeuralNetwork instance
    nn = SimpleNeuralNetwork(input_size=1, hidden_size=3, output_size=1, use_standard_scalar=True)
    
    # Train the network
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    st.success("Neural Network trained successfully!")

    # Store the trained network in session state
    st.session_state.nn = nn

# Check if we have a trained network
if 'nn' in st.session_state:
    nn = st.session_state.nn


    # Add a slider for the input
    nn_input = st.slider("Neural Network Input", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)


    # Make a prediction
    nn_output = nn.predict([nn_input])

    # Display the prediction
    st.write(f"Neural Network Output: {nn_output[0]:.4f}")

    # Create 3D visualization for Neural Network
    #x = np.linspace(-10, 10, 100)
    #y = np.linspace(-10, 10, 100)
    #X, Y = np.meshgrid(x, y)
    #Z_nn = np.zeros_like(X)

    #for i in range(X.shape[0]):
    #    for j in range(X.shape[1]):
    #        Z_nn[i, j] = nn.predict([X[i, j]])[0]

    #uirevision_params_nn = {
    #    'network_state': str(nn.hidden_layer) + str(nn.output_neuron)
    #}
    #st.plotly_chart(create_3d_graph(X, Y, Z_nn, "Neural Network Output Surface", uirevision_params_nn))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Neural Network Input-Output Relationship")
        # Create input data
        x = np.linspace(-10, 10, 200)
        # Get predictions
        y = np.array([nn.predict([xi]) for xi in x])
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlabel('Input', fontsize=12)
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title('Neural Network Input-Output Relationship', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        # Set axis limits
        ax.set_xlim(-10, 10)
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        # Clear the matplotlib figure to free up memory
        plt.clf()

    with col2:
        # Visualize the neural network structure
        st.subheader("Neural Network Structure")
        fig = draw_neural_network(nn)
        st.pyplot(fig)
        plt.clf()

else:
    st.info("Please train the neural network using the data table above.")

st.write("## How it works")
st.write("""
1. Adjust the number of inputs, input values, weights, and bias using the sliders above for both the single neuron and the MLP.
2. The single neuron's output is calculated based on the formula: output = activation(sum(inputs * weights) + bias).
3. The multi-layer perceptron (MLP) consists of an input layer, a hidden layer with 3 neurons, and an output layer with 1 neuron.
4. The MLP's output is calculated by passing the inputs through all layers, with each neuron in a layer receiving inputs from all neurons in the previous layer.
5. Two separate 3D surface plots show how the output changes with different input combinations for both the single neuron and the MLP (using only the first two inputs for visualization).
6. Experiment with different activation functions to see how they affect the behavior of both the single neuron and the MLP.
7. The network structures for both the single neuron and MLP are visualized using network graphs.
8. Use the "Test Single Neuron" and "Test Multi-Layer Perceptron" sections to input specific values and see the corresponding outputs.
""")

st.write("## Formulas")
st.latex(r"\text{Single Neuron: } output = f(\sum_{i=1}^n w_i x_i + b)")
st.latex(r"\text{MLP: } output = f_3(f_2(f_1(\sum_{i=1}^n w_{1i} x_i + b_1)))")
st.write("Where:")
st.write("- f is the activation function")
st.write("- w_i are the weights")
st.write("- x_i are the inputs")
st.write("- b is the bias")
st.write("- Subscripts 1, 2, 3 represent layers in the MLP")



