use crate::Matrix;
use crate::activations::Activations;
use rand;
use std::fmt;

// need to make z_matrix and a_Matrix private cuz right now the public

pub struct NeuralNetwork {
    learning_rate: f32,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    activations: Vec<Activations>,
    layers: Vec<usize>,
    z_matrix: Vec<Matrix>,
    a_matrix: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f32, activations: Vec<Activations>, layers: Vec<usize>) -> NeuralNetwork {
        let weights = Self::create_weights(&layers);
        let bias = Self::create_bias(&layers);
        if activations.len() != layers.len() - 1 {
            panic!("Activations length is {} and the number of layers is {}", activations.len(), layers.len());
        }
        let z_matrix: Vec<Matrix> = Vec::new();
        let a_matrix: Vec<Matrix> = Vec::new();
        NeuralNetwork {
            learning_rate, 
            weights, 
            bias,
            activations, 
            layers,
            z_matrix,
            a_matrix,
        }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn activations(&self) -> &[Activations] {
        &self.activations
    }

    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    pub fn bias(&self) -> &[Matrix] {
        &self.bias
    }

    pub fn weights(&self) -> &[Matrix] {
        &self.weights
    }

    pub fn z_matrix(&self) -> &[Matrix] {
        &self.z_matrix
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

    pub fn feed_forward(&mut self, input: &Matrix) -> Matrix {
        if input.rows() != self.layers[0] || input.cols() != 1 {
            panic!("Input dimensions mismatch: expected -> {}x1 and got {}x{}", self.layers[0], input.rows(), input.cols());
        }
        self.z_matrix.clear();
        self.a_matrix.clear();
        self.a_matrix.push(input.clone());
        for i in 0..self.weights.len() {
            let z = self.weights[i].mult(&self.a_matrix[i]).add(&self.bias[i]);
            self.z_matrix.push(z.clone());
            let a = z.apply(|x| self.activations[i].f(x));
            self.a_matrix.push(a);
        }
        self.a_matrix.last().unwrap().clone()
    }

    pub fn print_z_matrix(&self) {
        for vec in self.z_matrix.iter() {
            println!("{}", vec);
        }
    }

    pub fn print_a_matrix(&self) {
        for vec in self.a_matrix.iter() {
            println!("{}", vec);
        }
    }
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

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
        let activations = vec![
            Activations::Tanh, 
            Activations::Relu, 
            Activations::Sigmoid
        ];
        let nn = NeuralNetwork::new(learning_rate, activations.clone(), layers.clone());
        assert_eq!(nn.learning_rate(), learning_rate);
        assert_eq!(nn.layers(), &layers);
        assert_eq!(nn.activations(), &activations);
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
        let activations = vec![
            Activations::Relu, 
            Activations::Tanh, 
            Activations::Relu, 
            Activations::Sigmoid
        ];
        let nn = NeuralNetwork::new(learning_rate, activations, layers);
    }

    #[test]
    fn feed_forward_dimension_test() {
        let input = Matrix::new( 3, 1, vec![1.0, 2.0, 3.0]);
        let learning_rate: f32 = 0.01;
        let layers = vec![3, 1];
        let activations = vec![
            Activations::Sigmoid
        ];
        let mut nn = NeuralNetwork::new(learning_rate, activations.clone(), layers.clone());
        let output = nn.feed_forward(&input);
        assert_eq!(output.rows(), nn.layers().last().copied().unwrap());
        assert_eq!(output.cols(), 1);
    }

    #[test]
    #[should_panic]
    fn feed_forward_input_mismatch() {
        let input = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let learning_rate: f32 = 0.01;
        let layers = vec![3, 1];
        let activations = vec![
            Activations::Sigmoid
        ];
        let mut nn = NeuralNetwork::new(learning_rate, activations.clone(), layers.clone());
        let output = nn.feed_forward(&input);
    }
}