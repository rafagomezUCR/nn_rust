
mod matrix;
mod nn;
mod activations;

use crate::matrix::Matrix;
use crate::nn::NeuralNetwork;
use crate::activations::Activations;

fn main() {
    let input = Matrix::new(10, 1, vec![0.0; 10]);
    let activations = vec![Activations::Sigmoid, Activations::Relu, Activations::Relu];
    let layers = vec![10, 1, 1];
    let n = NeuralNetwork::new(0.03, input, activations, layers);
    n.feed_forward();
    println!("{}", n);
}
