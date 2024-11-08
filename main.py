import streamlit as st
import numpy as np
import plotly.graph_objects as go
from neuron import Neuron
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Interactive Neuron Visualization", layout="wide")

st.title("Interactive Neuron Visualization")

st.write("Use the controls below to modify the neuron's inputs, weights, and bias. The visualization will update in real-time.")

def create_2d_contribution_graph(weights, bias, activation_function, is_fig2=False):
    x = np.linspace(-5, 5, 200)
    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Calculate individual contributions
    contributions = []
    for i, w in enumerate(weights):
        y = x * w
        contributions.append(y)
        if not is_fig2:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=f'Input {i+1} Contribution',
                line=dict(color=colors[i % len(colors)])
            ))
    
    # Calculate total output with activation
    total_input = np.array([sum(c[i] for c in contributions) + bias for i in range(len(x))])
    neuron = Neuron(len(weights), activation_function)
    neuron.set_weights(weights)
    neuron.set_bias(bias)
    
    #total_output = np.array([neuron.forward([xi] + [0]*(len(weights)-1)) for xi in x])
    total_output = np.array([neuron._sigmoid(ti) for ti in total_input])
    #i = 0
    #print("Input       Total")
    #while i < 5:
    #    print(total_input[i], total_output[i])
    #    i = i + 1

    if is_fig2:
        fig.add_trace(go.Scatter(
            x=x, y=total_output,
            mode='lines',
            name='Total Output (with activation)',
            line=dict(color='white', width=2)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=total_input,
            mode='lines',
            name='Weighted Sums',
            line=dict(color='white', width=2)
        ))
    
    if is_fig2:
        fig.update_layout(
            title="Neuron Activation",
            xaxis_title="Input Value",
            yaxis_title="Output Value",
            showlegend=True,
            height=400,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor to the bottom of the graph
                y=-1.0,  # Adjust placement below the graph
                xanchor="center",  # Center the legend
                x=0.5  # Set it at the center of the x-axis
            )
        )
    else:
        fig.update_layout(
            title="Neuron Components and Output",
            xaxis_title="Input Value",
            yaxis_title="Output Value",
            showlegend=True,
            height=400,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor to the bottom of the graph
                y=-1.0,  # Adjust placement below the graph
                xanchor="center",  # Center the legend
                x=0.5  # Set it at the center of the x-axis
            )
        )
    
    return fig

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
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.subheader("Neuron Controls")
    num_inputs = st.slider("Number of Inputs", min_value=2, max_value=5, value=2, step=1)
    weights = [st.slider(f"Weight {i+1}", min_value=-2.0, max_value=2.0, value=1.0, step=0.1) for i in range(num_inputs)]
    bias = st.slider("Bias", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    activation_function = st.selectbox("Activation Function", ["Sigmoid", "ReLU", "Tanh"])
    inputs = [st.slider(f"Input {i+1}", min_value=-5.0, max_value=5.0, value=1.0, step=0.1) for i in range(num_inputs)]

    # Create neuron instance
    neuron = Neuron(num_inputs, activation_function)

    # Set neuron parameters
    neuron.set_weights(weights)
    neuron.set_bias(bias)

    # Calculate output
    output = neuron.forward(inputs)

    st.write(f"Single Neuron Output: {output:.4f}")
with col2:

    # Create 3D visualization for Single Neuron
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
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

    st.subheader("Test Single Neuron")
    input1 = st.number_input("Input 1", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    input2 = st.number_input("Input 2", min_value=-4.0, max_value=5.0, value=0.0, step=0.1)
    if st.button("Calculate Neuron Output"):
        test_inputs = [input1, input2] + inputs[2:]
        test_inputs = test_inputs[:neuron.num_inputs] + [0] * (neuron.num_inputs - len(test_inputs))
        test_output = neuron.forward(test_inputs)
        st.write(f"Neuron Output: {test_output:.4f}")

with col3:
    # Create placeholders for the graphs
    graph_placeholder1 = st.empty()
    graph_placeholder2 = st.empty()

    # Function to update graphs
    def update_graphs():
        fig1 = create_2d_contribution_graph(weights, bias, activation_function)
        graph_placeholder1.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_2d_contribution_graph(weights, bias, activation_function, True)
        graph_placeholder2.plotly_chart(fig2, use_container_width=True)

    # Initial graph drawing
    update_graphs()

    # Add a callback to update graphs when inputs change
    #for w in weights:
    #    w.on_change(lambda _: update_graphs())
    #bias.on_change(lambda _: update_graphs())


#st.subheader("Neuron Output")
# Display Single Neuron Structure
#st.plotly_chart(create_network_graph([num_inputs, 1]))
                    



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
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Add bias annotations
    for i, node in enumerate(["I1", "H1", "H2", "H3", "O1"]):
        if node == "I1":
            bias = 0
        elif node == "O1":
            bias = nn.output_neuron.bias
        else:
            bias = nn.hidden_layer[i-1].bias
        plt.annotate(f"b: {bias:.2f}", xy=pos[node], xytext=(5, 5), 
                     textcoords="offset points", fontsize=8)

    plt.title("Neural Network Structure with Weights and Biases")
    plt.axis('off')
    return plt

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



