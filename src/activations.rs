

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