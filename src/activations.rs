
#[derive(Debug)]
pub enum Activations {
    Sigmoid,
    Tanh,
    Relu,
}

impl Activations {
    pub fn f(&self, x: f64) -> f64 {
        match self {
            Activations::Sigmoid => {
                1.0 / (1.0 + (-x).exp())
            }
            Activations::Tanh => {
                x.tanh()
            }
            Activations::Relu => {
                x.max(0.0)
            }
        }
    }

    pub fn df(&self, x: f64) -> f64 {
        match self {
            Activations::Sigmoid => {
                let s = self.f(x);
                s * (1.0 - s)
            }
            Activations::Tanh => {
                let s = x.tanh();
                1.0 - s * s
            }
            Activations::Relu => {
                if x <= 0.0 { 0.0 } else { 1.0 }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    const THRESHOLD: f64 = 1e-9;

    #[test]
    fn sigmoid_f_test() {
        let expected = 1.0 / (1.0 + (-1.0f64).exp());
        assert!( (Activations::Sigmoid.f(0.0) - 0.5).abs() < THRESHOLD);
        assert!( (Activations::Sigmoid.f(1.0) - expected).abs() < THRESHOLD)
    }
}