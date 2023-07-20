use core::panic;

use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Generate a zeroz matrix
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// Generate a random matrix
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();

        // create a matrix of zeros
        let mut res = Matrix::zeros(rows, cols);

        // randomize the values of the res matrix
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0
            }
        }

        // return the random matrix
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    /// myltiply two matrix
    pub fn multiply(&mut self, other: &Matrix) -> Matrix {
        // check if the multiplication is possible between the two matrix
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, other.cols);

        // multiply the two matrix
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }

        // return the result matrix
        res
    }

    /// Add two matrix
    pub fn add(&mut self, other: &Matrix) -> Matrix {
        // check if the addition is possible between the two matrix
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to add by matrix of incorrect dimensions");
        }

        // add the two matrix
        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        // return the result matrix
        res
    }

    pub fn substract(&mut self, other: &Matrix) -> Matrix {
        // check if the substraction is possible between the two matrix
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to substract by matrix of incorrect dimensions");
        }

        // substract the other matrix to the actual one
        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        // return the result matrix
        res
    }

    // map a function to edit the matrix
    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect(),
        )
    }

    /// transpose the matrix
    pub fn transpose(&mut self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);

        // transpose into the res matrix
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        // return the result matrix
        res
    }

    pub fn print_size(&self, name: &str) -> () {
        println!(
            "{} matrix size: {}x{}",
            name,
            self.data.len(),
            self.data[0].len()
        );
    }
}
