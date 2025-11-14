use std::fmt;
use std::ops::Add;

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        // should add some safety by either panicing or returning result
        // or by creating a try_new func that returns a result and new returns the unwrap
        Matrix{rows, cols, data}
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

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

    pub fn add(&self, rhs: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            return Err("Trying to Add Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x + y).collect();
        Ok(Matrix{ rows: self.rows(), cols: self.cols(), data})
    }

    pub fn subtract(&self, rhs: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            return Err("Trying to subtract Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x - y).collect();
        Ok(Matrix{rows: self.rows(), cols: self.cols(), data})
    }

    pub fn elementwise_mult(&self, rhs: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            return Err("Trying to do Elementwise Multiplication on Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x * y).collect();
        Ok(Matrix{rows: self.rows(), cols: self.cols(), data})
    }

    pub fn mult(&self, rhs: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols() != rhs.rows() {
            return Err("Left Matrix columns don't match Right Matrix rows!");
        }
        let mut data = vec![0.0; self.rows() * rhs.cols()];
        let self_data = self.data();
        let self_rows = self.rows();
        let self_cols = self.cols();
        let rhs_data = rhs.data();
        let rhs_cols = rhs.cols();
        for i in 0..self_rows {
            for j in 0..rhs_cols {
                let mut sum = 0.0;
                for k in 0..self_cols {
                    sum += self_data[i * self_cols + k] * rhs_data[k * rhs_cols + j];
                }
                data[i * rhs_cols + j] = sum;
            }
        }
        Ok(Matrix{rows: self_rows, cols: rhs_cols, data})
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            panic!("Trying to Add Matrices of Different Dimensions!");
        }
        let data: Vec<f64> = self.data().iter().zip(rhs.data()).map(|(x, y)| x + y).collect();
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