use crate::Matrix;
use crate::activations::Activations;

pub struct NeuralNetwork {
    learning_rate: f32,
    data: Matrix,
    weights: Matrix,
    bias: Matrix,
    activations: Vec<Activations>,
    layers: Vec<usize>,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f32, data: Matrix, activations: Vec<Activations>, layers: Vec<usize>) -> NeuralNetwork {
        let weights = Self::create_weights();
        let bias = Self::create_bias();
        NeuralNetwork {
            learning_rate, 
            data,
            weights, 
            bias,
            activations, 
            layers,
        }
    }

    fn create_weights() -> Matrix {
        todo!()
    }

    fn create_bias() -> Matrix {
        todo!()
    }
}