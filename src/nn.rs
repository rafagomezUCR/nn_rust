use crate::Matrix;
use crate::activations::Activations;
use rand;
use std::fmt;

pub struct NeuralNetwork {
    learning_rate: f32,
    input: Matrix,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    activations: Vec<Activations>,
    layers: Vec<usize>,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f32, input: Matrix, activations: Vec<Activations>, layers: Vec<usize>) -> NeuralNetwork {
        let weights = Self::create_weights(&layers);
        let bias = Self::create_bias(&layers);
        NeuralNetwork {
            learning_rate, 
            input,
            weights, 
            bias,
            activations, 
            layers,
        }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn activations(&self) -> &Vec<Activations> {
        &self.activations
    }

    pub fn layers(&self) -> &Vec<usize> {
        &self.layers
    }

    pub fn bias(&self) -> &Vec<Matrix> {
        &self.bias
    }

    pub fn weights(&self) -> &Vec<Matrix> {
        &self.weights
    }

    pub fn input(&self) -> &Matrix {
        &self.input
    }

    fn create_weights(layers: &[usize]) -> Vec<Matrix> {
        let mut weights = Vec::new();
        for i in 0..(layers.len() - 1) {
            let rows = layers[i + 1];
            let cols = layers[i];
            let weight_data = (0..rows * cols).map(|_| rand::random_range(-1.0..1.0)).collect();
            weights.push(Matrix::new(rows, cols, weight_data));
        }
        weights
    }

    fn create_bias(layers: &[usize]) -> Vec<Matrix> {
        layers.iter().map(|&x| {
            let bias = vec![0.01; x];
            Matrix::new(x, 1, bias)
        }).collect()
    }
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "input: {}", self.input())?;
        writeln!(f, "learning rate: {}", self.learning_rate())?;
        
        writeln!(f, "activations:")?;
        for (i, a) in self.activations().iter().enumerate() {
            writeln!(f, "   layer {}: {:?}", i, a)?;
        }
        
        writeln!(f, "layers: [{}]", self.layers().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))?;
        
        writeln!(f, "biases:")?;
        for (i, b) in self.bias().iter().enumerate() {
            writeln!(f, "   layer {}: {}", i, b)?;
        }
        
        writeln!(f, "weights:")?;
        for (i, w) in self.weights().iter().enumerate() {
            writeln!(f, "   layer {}: {}", i, w)?;
        }
        
        Ok(())
    }
}