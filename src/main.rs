
mod matrix;

use crate::matrix::Matrix;

fn main() {
    let v: Vec<f64> = vec![1.0,1.0,1.0,1.0,1.0,1.0];
    let c: Vec<f64> = vec![1.0,1.0,1.0,1.0,1.0,1.0];
    let a = Matrix::new(2, 3, v);
    let b = Matrix::new(2, 3, c);
    println!("{}", a);
    println!("{}", b);
    let c = a.add(&b);
    println!("{}", c.unwrap());
}
