

enum Activations {
    Sigmoid(fn(f64) -> f64),
    Tanh(fn(f64) -> f64),
    Relu(fn(f64) -> f64),
}

pub struct Activation {
    pub func: fn(&[f64]),
    pub derivative: fn(&[f64]),
}

pub struct NeuralNetwork {
    learning_rate: f32;
    data: Matrix;
    weights: Matrix;
    bias: Matrix;
    activations: Vec<Activations>,
    layers: Vec<usize>,
}