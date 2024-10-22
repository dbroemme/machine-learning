import streamlit as st
import numpy as np
import plotly.graph_objects as go
from neuron import Neuron
from mlp import create_mlp_with_random_params, MLP

st.set_page_config(page_title="Interactive Neuron Visualization", layout="wide")

st.title("Interactive Neuron Visualization")

st.write("Use the controls below to modify the neuron's inputs, weights, and bias. The visualization will update in real-time.")

def create_network_graph(layer_sizes, is_mlp=False):
    edges = []
    node_x = []
    node_y = []
    node_text = []
    
    for i, layer_size in enumerate(layer_sizes):
        layer_x = i
        for j in range(layer_size):
            node_y.append(j - (layer_size - 1) / 2)
            node_x.append(layer_x)
            node_text.append(f"{'Neuron' if i > 0 else 'Input'} {j+1}")
            
            if i < len(layer_sizes) - 1:
                for k in range(layer_sizes[i + 1]):
                    edges.append((layer_x, j - (layer_size - 1) / 2, layer_x + 1, k - (layer_sizes[i + 1] - 1) / 2))

    edge_x = []
    edge_y = []
    for edge in edges:
        edge_x.extend([edge[0], edge[2], None])
        edge_y.extend([edge[1], edge[3], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            size=10,
            color=[],
            line_width=2))

    node_trace.text = node_text
    
    title = "Multi-Layer Perceptron Structure" if is_mlp else "Single Neuron Structure"
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

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

# Single Neuron Section
st.header("Single Neuron")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Neuron Controls")
    num_inputs = st.slider("Number of Inputs", min_value=2, max_value=5, value=2, step=1)
    inputs = [st.slider(f"Input {i+1}", min_value=-10.0, max_value=10.0, value=1.0, step=0.1) for i in range(num_inputs)]
    weights = [st.slider(f"Weight {i+1}", min_value=-10.0, max_value=10.0, value=1.0, step=0.1) for i in range(num_inputs)]
    bias = st.slider("Bias", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    activation_function = st.selectbox("Activation Function", ["Sigmoid", "ReLU", "Tanh"])

    # Create neuron instance
    neuron = Neuron(num_inputs, activation_function)

    # Set neuron parameters
    neuron.set_weights(weights)
    neuron.set_bias(bias)

    # Calculate output
    output = neuron.forward(inputs)

with col2:
    st.subheader("Neuron Output")
    st.write(f"## Single Neuron Output: {output:.4f}")
    
    # Display Single Neuron Structure
    st.plotly_chart(create_network_graph([num_inputs, 1]))

# Create 3D visualization for Single Neuron
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z_neuron = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        viz_inputs = [X[i, j], Y[i, j]]
        full_inputs = viz_inputs + inputs[2:]
        full_inputs = full_inputs[:neuron.num_inputs] + [0] * (neuron.num_inputs - len(full_inputs))
        Z_neuron[i, j] = neuron.forward(full_inputs)

uirevision_params_neuron = {
    'weights': weights,
    'bias': bias,
    'activation': activation_function
}
st.plotly_chart(create_3d_graph(X, Y, Z_neuron, "Single Neuron Output Surface", uirevision_params_neuron))

# New section: Form for Single Neuron input
st.subheader("Test Single Neuron")
col1, col2 = st.columns(2)

with col1:
    input1 = st.number_input("Input 1", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    input2 = st.number_input("Input 2", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    if st.button("Calculate Neuron Output"):
        test_inputs = [input1, input2] + inputs[2:]
        test_inputs = test_inputs[:neuron.num_inputs] + [0] * (neuron.num_inputs - len(test_inputs))
        test_output = neuron.forward(test_inputs)
        st.write(f"Neuron Output: {test_output:.4f}")

# Multi-Layer Perceptron (MLP) Section
st.header("Multi-Layer Perceptron (MLP)")

# Create MLP instance
mlp_layer_sizes = [num_inputs, 3, 1]
mlp = create_mlp_with_random_params(mlp_layer_sizes, activation_function)

st.subheader("MLP Controls")
mlp_weights = []
mlp_biases = []

for i, layer in enumerate(mlp.layers):
    st.write(f"Layer {i+1}")
    layer_weights = []
    layer_biases = []
    for j, neuron in enumerate(layer):
        col1, col2 = st.columns(2)
        with col1:
            neuron_weights = [st.slider(f"L{i+1}N{j+1} Weight {k+1}", min_value=-10.0, max_value=10.0, value=float(w), step=0.1) for k, w in enumerate(neuron.weights)]
        with col2:
            neuron_bias = st.slider(f"L{i+1}N{j+1} Bias", min_value=-10.0, max_value=10.0, value=float(neuron.bias), step=0.1)
        layer_weights.append(neuron_weights)
        layer_biases.append(neuron_bias)
    mlp_weights.append(layer_weights)
    mlp_biases.append(layer_biases)

# Set MLP weights and biases
mlp.set_weights_and_biases(mlp_weights, mlp_biases)

# Calculate MLP output
mlp_output = mlp.forward(inputs)

st.write(f"## Multi-Layer Perceptron Output: {mlp_output:.4f}")

# Display MLP Structure
st.plotly_chart(create_network_graph(mlp_layer_sizes, is_mlp=True))

# Create 3D visualization for MLP
Z_mlp = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        viz_inputs = [X[i, j], Y[i, j]]
        full_inputs = viz_inputs + inputs[2:]
        full_inputs = full_inputs[:mlp.layers[0][0].num_inputs] + [0] * (mlp.layers[0][0].num_inputs - len(full_inputs))
        Z_mlp[i, j] = mlp.forward(full_inputs)

uirevision_params_mlp = {
    'weights': mlp_weights,
    'biases': mlp_biases,
    'activation': activation_function
}
st.plotly_chart(create_3d_graph(X, Y, Z_mlp, "MLP Output Surface", uirevision_params_mlp))

# New section: Form for MLP input
st.subheader("Test Multi-Layer Perceptron")
col1, col2 = st.columns(2)

with col1:
    mlp_input1 = st.number_input("MLP Input 1", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    mlp_input2 = st.number_input("MLP Input 2", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    if st.button("Calculate MLP Output"):
        mlp_test_inputs = [mlp_input1, mlp_input2] + inputs[2:]
        mlp_test_inputs = mlp_test_inputs[:mlp.layers[0][0].num_inputs] + [0] * (mlp.layers[0][0].num_inputs - len(mlp_test_inputs))
        mlp_test_output = mlp.forward(mlp_test_inputs)
        st.write(f"MLP Output: {mlp_test_output:.4f}")

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