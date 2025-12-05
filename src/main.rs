
mod matrix;
mod nn;
mod activations;

use crate::matrix::Matrix;
use crate::nn::NeuralNetwork;
use crate::activations::Activations;

fn main() {
    let input = Matrix::new(5, 1, vec![0.1; 5]);
    let activations = vec![Activations::Relu, Activations::Sigmoid];
    let layers = vec![5, 3, 1];
    let mut n = NeuralNetwork::new(0.03, activations, layers);
    let a = n.feed_forward(&input);
    //println!("{}", n);
    //n.print_z_matrix();
    println!("{}", a);
}
