

enum Activations {
    Sigmoid,
    Tanh,
    Relu,
}

pub struct ActivationDefinition {
    func: fn(&f64) -> f64,
    derivative: fn(&f64) -> f64,
}

pub const Activation: ActivationDefinition = ActivationDefinition {
    func: match Activations
    derivative: 
};