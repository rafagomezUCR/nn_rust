use std::fmt;
use std::error::Error;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Matrix {
            rows,
            cols,
            data,
        }
    }

    pub fn add(&self, right_matrix: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        // check for different sizes
        let rows = self.rows;
        let cols = self.cols;
        let mut data = vec![0.0; self.data.len()];
        for row in 0..rows {
            for col in 0..cols {
                data[row * cols + col] = self.data[row * cols + col] + right_matrix.data[row * cols + col];
            }
        }
        Ok(Matrix{ rows, cols, data })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "rows: {}", self.rows)?;
        writeln!(f, "cols: {}", self.cols)?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.data[row * self.cols + col];
                write!(f, "{:<8.3}", val)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}