
mod matrix;

use crate::matrix::Matrix;

fn main() {
    let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0];
    let a = Matrix::new(2, 2, v);
    let b = Matrix::new(3, 1, y);
    let c = &a + &b;
    println!("{}", c);
}
