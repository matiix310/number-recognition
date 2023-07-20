use crate::matrix::matrix::Matrix;

use super::activations;

#[derive(Clone)]
pub struct Layer<'a> {
    pub weights: Matrix,
    pub biases: Matrix,
    pub size_in: usize,
    pub size_out: usize,
    activation: activations::Activation<'a>,
    inputs: Vec<f64>,
    data: Matrix,
    cost_gradient_w: Matrix,
    cost_gradient_b: Matrix,
}

impl Layer<'_> {
    pub fn new(size_in: usize, size_out: usize, activation: activations::Activation) -> Layer {
        Layer {
            weights: Matrix::random(size_out, size_in),
            biases: Matrix::random(size_out, 1),
            size_in,
            size_out,
            activation,
            inputs: vec![],
            data: Matrix::zeros(0, 0),
            cost_gradient_w: Matrix::zeros(size_out, size_in),
            cost_gradient_b: Matrix::zeros(size_out, 1),
        }
    }

    pub fn compute_output(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.inputs = inputs.clone();
        self.data = self
            .weights
            .multiply(&Matrix::from(vec![inputs]).transpose())
            .add(&self.biases);

        self.data.map(self.activation.function).transpose().data[0].clone()
    }

    #[allow(unused)]
    pub fn node_cost(output: f64, target: f64) -> f64 {
        (output - target).powi(2)
    }

    pub fn node_cost_derivative(output: f64, target: f64) -> f64 {
        2.0 * (output - target)
    }

    pub fn apply_gradients(&mut self, learning_rate: &f64) {
        self.biases = self
            .biases
            .substract(&self.cost_gradient_b.map(&|x| x * learning_rate));
        self.weights = self
            .weights
            .substract(&self.cost_gradient_w.map(&|x| x * learning_rate));
        // TODO ???? wtf no substraction ????
    }

    pub fn get_output_layer_node_value(&mut self, target: &Vec<f64>) -> Vec<f64> {
        let mut node_values = vec![];
        let activation_output = &self.data.map(self.activation.function).transpose().data[0];
        let weighted_output = &self.data.transpose().data[0];

        for i in 0..self.size_out {
            let cost_derivative = Layer::node_cost_derivative(activation_output[i], target[i]);
            let activation_derivative = (self.activation.derivative)(weighted_output[i]);
            node_values.push(activation_derivative * cost_derivative);
        }

        node_values
    }

    pub fn get_hidden_layer_node_value(
        &mut self,
        old_layer_weights: &Matrix,
        old_node_values: &Vec<f64>,
    ) -> Vec<f64> {
        let mut node_values = vec![];

        for i in 0..self.size_out {
            let mut node_value = 0.0;
            for j in 0..old_node_values.len() {
                node_value += old_layer_weights.data[i][j] * old_node_values[j];
            }
            node_value *= (self.activation.derivative)(self.data.data[i][0]);
            node_values.push(node_value);
        }

        node_values
    }

    pub fn update_gradients(&mut self, node_values: &Vec<f64>) {
        for i in 0..self.size_out {
            for j in 0..self.size_in {
                self.cost_gradient_w.data[i][j] += self.inputs[j] * node_values[i];
            }
            self.cost_gradient_b.data[i][0] += node_values[i];
        }
    }

    pub fn clear_gradients(&mut self) {
        self.cost_gradient_w = Matrix::zeros(self.size_out, self.size_in);
        self.cost_gradient_b = Matrix::zeros(self.size_out, 1);
    }
}
