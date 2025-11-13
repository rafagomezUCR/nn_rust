use std::fmt;

pub struct Matrix {
    rows: usize,
    cols: usize,
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