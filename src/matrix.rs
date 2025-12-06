use std::fmt;
use std::ops::{Add, Sub, Mul};

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        if rows * cols != data.len() {
            panic!("Matrix::new received mismatched dimensions and data length!");
        }
        Matrix{ rows, cols, data}
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn dim_check(&self) {
        println!("dim: {}x{}", self.rows(), self.cols());
    }

    pub fn apply<F>(&self, func: F) -> Matrix where F: Fn(f64) -> f64 {
        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: self.data().iter().map(|&x| func(x)).collect(),
        }
    }

    pub fn transpose(&self) -> Matrix {
        let (rows, cols) = (self.rows(), self.cols());
        let mut new_data: Vec<f64> = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                new_data[col * rows + row] = self.data()[row * cols + col];
            }
        }
        Matrix{
            rows: self.cols,
            cols: self.rows,
            data: new_data,
        }
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to Add Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x + y).collect();
        Matrix{ rows: self.rows(), cols: self.cols(), data}
    }

    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to subtract Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x - y).collect();
        Matrix{rows: self.rows(), cols: self.cols(), data}
    }

    pub fn elementwise_mult(&self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to do Elementwise Multiplication on Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x * y).collect();
        Matrix{rows: self.rows(), cols: self.cols(), data}
    }

    pub fn mult(&self, rhs: &Matrix) -> Matrix {
        if self.cols() != rhs.rows() {
            panic!("Left Matrix columns don't match Right Matrix rows!\n {}x{} {}x{}", self.rows(), self.cols(), rhs.rows(), rhs.cols());
        }
        let mut data = vec![0.0; self.rows() * rhs.cols()];
        let (self_data, self_rows, self_cols) = (self.data(), self.rows(), self.cols());
        let (rhs_data, rhs_cols) = (rhs.data(), rhs.cols());
        for i in 0..self_rows {
            for j in 0..rhs_cols {
                let mut sum = 0.0;
                for k in 0..self_cols {
                    sum += self_data[i * self_cols + k] * rhs_data[k * rhs_cols + j];
                }
                data[i * rhs_cols + j] = sum;
            }
        }
        Matrix{rows: self_rows, cols: rhs_cols, data}
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to Add Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x + y).collect();
        Matrix{ rows: self.rows(), cols: self.cols(), data}
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to Subtract Matrices of Different Dimensions");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x - y).collect();
        Matrix{ rows: self.rows(), cols: self.cols(), data}
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to do Elementwise Multiplication on Matrices of Different Dimensions!");
        }
        let data = self.data().iter().zip(rhs.data()).map(|(x, y)| x * y).collect();
        Matrix{ rows: self.rows(), cols: self.cols(), data}
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(data: Vec<f64>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: 1,
            data,
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{} x {} Matrix", self.rows(), self.cols())?;
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let val = self.data()[row * self.cols() + col];
                write!(f, "{:<8.3}", val)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrix_test() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m, Matrix {rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0]});
    }

    #[test]
    #[should_panic]
    fn new_matrix_mismatch_test() {
        Matrix::new(2, 2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn transpose_test() {
        let m = Matrix::new(3, 5, vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ]);
        assert_eq!(m.transpose(), Matrix{
            rows: 5, 
            cols: 3, 
            data: vec![
                1.0, 6.0, 11.0,
                2.0, 7.0, 12.0,
                3.0, 8.0, 13.0,
                4.0, 9.0, 14.0,
                5.0, 10.0, 15.0
            ]
        })
    }

    #[test]
    fn add_success_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(&a + &b, c);
        assert_eq!(a.add(&b), c);
    }

    #[test]
    #[should_panic]
    fn add_mismatch_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        &a + &b;
    }

    #[test]
    fn subtract_success_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = Matrix::new(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&a - &b, c);
        assert_eq!(a.sub(&b), c);
    }

    #[test]
    #[should_panic]
    fn subtract_mismatch_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        &a - &b;
    }

    #[test]
    fn elementwise_mult_success_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = Matrix::new(2, 2, vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(&a * &b, c);
        assert_eq!(a.elementwise_mult(&b), c);
    }

    #[test]
    #[should_panic]
    fn elementwise_mult_mismatch_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        &a * &b;
    }

    #[test]
    fn mult_success_test() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = Matrix::new(2, 2, vec![22.0, 28.0, 49.0, 64.0]);
        assert_eq!(a.mult(&b), c);
    }

    #[test]
    #[should_panic]
    fn mult_mismatch_test() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.mult(&b);
    }

    #[test]
    fn from_vec_test() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let vec_len = vec.len();
        let m = Matrix::from(vec);
        assert_eq!(m.rows(), vec_len);
        assert_eq!(m.cols(), 1);
    }

    #[test]
    fn matrix_display_test() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let my_string_string: String = format!("{}", &m);
        let expected_string: String = String::from("2 x 2 Matrix\n1.000   2.000   \n3.000   4.000   \n");
        assert_eq!(my_string_string, expected_string);
    }
}