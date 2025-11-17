
mod matrix;
mod nn;
mod activations;

use crate::matrix::Matrix;
use crate::nn::NeuralNetwork;

fn main() {
    let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let a = Matrix::new(2, 2, v);
    let b = Matrix::new(2, 2, y);
    let c = &a * &b;
    println!("{}", c);
}
