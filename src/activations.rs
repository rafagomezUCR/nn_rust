
#[derive(Debug, PartialEq, Clone)]
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
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let expected = 1.0 / (1.0 + (-v).exp());
            let actual = Activations::Sigmoid.f(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }

    #[test]
    fn sigmoid_df_test() {
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let s = 1.0 / (1.0 + (-v).exp());
            let expected = s * (1.0 - s);
            let actual = Activations::Sigmoid.df(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }

    #[test]
    fn tanh_f_test() {
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let expected = v.tanh();
            let actual = Activations::Tanh.f(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }

    #[test]
    fn tanh_df_test() {
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let expected = 1.0 - v.tanh() * v.tanh();
            let actual = Activations::Tanh.df(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }

    #[test]
    fn relu_f_test() {
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let expected = v.max(0.0);
            let actual = Activations::Relu.f(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }

    #[test]
    fn relu_df_test() {
        let values: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
        for &v in values.iter() {
            let expected = {
                if v <= 0.0 { 0.0 } else { 1.0 }
            };
            let actual = Activations::Relu.df(v);
            assert!( (actual - expected).abs() < THRESHOLD);
        }
    }
}