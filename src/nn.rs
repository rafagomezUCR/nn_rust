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
        if activations.len() != layers.len() {
            panic!("Activations length is {} and the number of layers is {}", activations.len(), layers.len());
        }
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
        layers.iter().skip(1).map(|&x| {
            let bias = vec![0.01; x];
            Matrix::new(x, 1, bias)
        }).collect()
    }

    pub fn feed_forward() -> Matrix {
        Matrix::new(1, 1, vec![1.0])
    }
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "input data")?;
        writeln!(f, "----------")?;
        writeln!(f, "{} x {} Matrix", self.input().rows(), self.input().cols())?;

        writeln!(f, "\nlearning rate: {}", self.learning_rate())?;
        
        writeln!(f, "\nactivations")?;
        writeln!(f, "-----------")?;
        for (i, a) in self.activations().iter().enumerate() {
            writeln!(f, "layer {}: {:?}", i, a)?;
        }
        
        writeln!(f, "\nlayers: [{}]", self.layers().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))?;

        writeln!(f, "\nbias")?;
        writeln!(f, "----")?;
        for (i, b) in self.bias().iter().enumerate() {
            writeln!(f, "layer {}: {} x {}", i, b.rows(), b.cols())?;
        }
        
        writeln!(f, "\nweights:")?;
        writeln!(f, "--------")?;
        for (i, w) in self.weights().iter().enumerate() {
            writeln!(f, "layer {}: {} x {}", i, w.rows(), w.cols())?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_nn_and_getters_test() {
        let learning_rate: f32 = 0.01;
        let layers = vec![5, 3, 2, 1];
        let input = Matrix::new(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let activations = vec![
            Activations::Relu, 
            Activations::Tanh, 
            Activations::Relu, 
            Activations::Sigmoid
        ];
        let nn = NeuralNetwork::new(learning_rate, input.clone(), activations.clone(), layers.clone());
        assert_eq!(nn.learning_rate(), learning_rate);
        assert_eq!(nn.layers(), &layers);
        assert_eq!(nn.activations(), &activations);
        assert_eq!(nn.input(), &input);
        for (i, w_layer) in nn.weights().iter().enumerate() {
            assert_eq!(w_layer.rows(), layers[i + 1]);
            assert_eq!(w_layer.cols(), layers[i]);
            for w in w_layer.data() {
                assert!(*w <= 1.0 && *w >= -1.0);
            }
        }
        for (i, b_layer) in nn.bias().iter().enumerate() {
            assert_eq!(b_layer.rows(), layers[i + 1]);
            assert_eq!(b_layer.cols(), 1);
            for b in b_layer.data(){
                assert_eq!(*b, 0.01);
            }
        }
    }

    #[test]
    #[should_panic]
    fn nn_layer_mismatch() {
        let learning_rate: f32 = 0.01;
        let layers = vec![5, 3, 2];
        let input = Matrix::new(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let activations = vec![
            Activations::Relu, 
            Activations::Tanh, 
            Activations::Relu, 
            Activations::Sigmoid
        ];
        let nn = NeuralNetwork::new(learning_rate, input, activations, layers);
    }
}