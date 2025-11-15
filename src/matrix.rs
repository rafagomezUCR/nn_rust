use std::fmt;
use std::ops::{Add, Sub, Mul};

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
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

    pub fn transpose(mut self) -> Self {
        let (rows, cols) = (self.rows(), self.cols());
        let mut new_data: Vec<f64> = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                new_data[col * rows + row] = self.data()[row * cols + col];
            }
        }
        self.rows = cols;
        self.cols = rows;
        self.data = new_data;
        self
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to Add Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x + y).collect();
        Matrix{ rows: self.rows(), cols: self.cols(), data}
    }

    pub fn subtract(&self, rhs: &Matrix) -> Matrix {
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
            panic!("Left Matrix columns don't match Right Matrix rows!");
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

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "rows: {}", self.rows())?;
        writeln!(f, "cols: {}", self.cols())?;
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