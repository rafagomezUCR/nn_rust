
mod matrix;

use crate::matrix::Matrix;

fn main() {
    let v: Vec<f64> = vec![1.0,2.0,3.0,4.0,4.0,5.0,34.0,2.02,23.0,3.0,23.0,4.0];
    let a = Matrix::new(3, 3, v);
    println!("{}", a);
}
