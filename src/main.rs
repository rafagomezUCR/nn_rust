
mod matrix;
mod nn;
mod activations;

use crate::matrix::Matrix;
use crate::nn::NeuralNetwork;
use crate::activations::Activations;
use std::collections::VecDeque;

fn main() {
    // XOR dataset
    let inputs = vec![
        Matrix::from(vec![0.0, 0.0]),
        Matrix::from(vec![0.0, 1.0]),
        Matrix::from(vec![1.0, 0.0]),
        Matrix::from(vec![1.0, 1.0]),
    ];
    let outputs = vec![
        Matrix::from(vec![0.0]),
        Matrix::from(vec![1.0]),
        Matrix::from(vec![1.0]),
        Matrix::from(vec![0.0]),
    ];

    // Neural network with 2 inputs, 2 hidden neurons, 1 output
    let activations = vec![Activations::Tanh, Activations::Sigmoid];
    let layers = vec![2, 2, 1];
    let mut nn = NeuralNetwork::new(0.1, activations, layers);

    // Training loop
    let epochs = 1500;
    for epoch in 0..epochs {
        let mut loss = 0.0;
        for (x, y) in inputs.iter().zip(outputs.iter()) {
            let output = nn.feed_forward(x);
            let diff = output.sub(y);
            loss += diff.data().iter().map(|v| v*v).sum::<f64>();
            nn.backpropagation(y);
        }
        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss);
        }
    }

    // Test trained network
    println!("\nTesting XOR:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let output = nn.feed_forward(x);
        println!("Input: {:?}, Output: {:.4}, Target: {:.0}", x.data(), output.data()[0], y.data()[0]);
    }
}



