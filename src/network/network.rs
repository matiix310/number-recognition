use std::{
    fs::File,
    io::{Read, Write},
};

use progress_bar::*;

use crate::matrix::matrix::Matrix;

use super::{
    activations::{self, Activation},
    layer::Layer,
};

#[derive(Clone)]
pub struct Network<'a> {
    layers: Vec<Layer<'a>>,
    layer_count: usize,
    learning_rate: f64,
}

impl Network<'_> {
    pub fn new<'a>(
        layers_struct: &'a Vec<usize>,
        learning_rate: &'a f64,
        activation_function: Activation<'a>,
    ) -> Network<'a> {
        let mut layers = Vec::<Layer>::new();

        for i in 1..layers_struct.len() {
            layers.push(Layer::new(
                layers_struct[i - 1],
                layers_struct[i],
                activation_function.clone(),
            ))
        }

        Network {
            layers,
            layer_count: layers_struct.len() - 1,
            learning_rate: learning_rate.clone(),
        }
    }

    pub fn load_from_file<'a>(
        path: &str,
        learning_rate: &f64,
        activation_function: Activation<'a>,
    ) -> Network<'a> {
        let mut file = File::open(path).expect("Can't open the model file");

        // layer_count
        let mut layer_count_buf: [u8; 2] = [0, 2];
        file.read_exact(&mut layer_count_buf)
            .expect("Can't read the layer_count");
        let layer_count = u16::from_be_bytes(layer_count_buf) as usize - 1;

        // layers_size
        let mut layers = Vec::<Layer>::new();
        let mut layers_size: Vec<usize> = vec![0];
        for _ in 0..layer_count + 1 {
            let mut layer_size_buf: [u8; 2] = [0; 2];
            file.read_exact(&mut layer_size_buf)
                .expect("Can't read the layer size");
            let size_out = u16::from_be_bytes(layer_size_buf) as usize;
            let layer = Layer::new(
                layers_size.last().expect("No precedent layer").clone(),
                size_out,
                activation_function.clone(),
            );
            layers_size.push(size_out);
            layers.push(layer);
        }

        layers.remove(0);

        // weights and biases for each layer
        for layer_index in 0..layer_count {
            let mut weights =
                Matrix::zeros(layers[layer_index].size_out, layers[layer_index].size_in);
            for i in 0..layers[layer_index].size_out {
                for j in 0..layers[layer_index].size_in {
                    let mut weight_buf: [u8; 8] = [0; 8];
                    file.read_exact(&mut weight_buf)
                        .expect("Can't read the weights");
                    weights.data[i][j] = f64::from_be_bytes(weight_buf);
                }
            }
            let mut biases = Matrix::zeros(layers[layer_index].size_out, 1);
            for i in 0..layers[layer_index].size_out {
                let mut bias_buf: [u8; 8] = [0; 8];
                file.read_exact(&mut bias_buf)
                    .expect("Can't read the biases");
                biases.data[i][0] = f64::from_be_bytes(bias_buf);
            }
            layers[layer_index].weights = weights;
            layers[layer_index].biases = biases;
        }

        Network {
            layer_count,
            layers,
            learning_rate: learning_rate.clone(),
        }
    }

    pub fn feed_forwards(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut current: Vec<f64> = input.clone();
        for layer_index in 0..self.layers.len() {
            current = self.layers[layer_index].compute_output(current);
        }
        current
    }

    pub fn train_with_batch(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &Vec<Vec<f64>>,
        epochs: usize,
        batch_size: usize,
    ) -> () {
        println!(
            "Splitting the data into batches of {} elements...",
            batch_size
        );
        // create all the batches
        let mut data: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = Vec::new();

        for i in 0..inputs.len() / batch_size {
            let mut batch_inputs: Vec<Vec<f64>> = Vec::new();
            let mut batch_targets: Vec<Vec<f64>> = Vec::new();
            for j in 0..batch_size {
                batch_inputs.push(inputs[i * batch_size + j].clone());
                batch_targets.push(targets[i * batch_size + j].clone());
            }
            data.push((batch_inputs, batch_targets));
        }

        println!("Starting the learnig process...");
        for i in 0..epochs {
            let batch_index = i % data.len();
            self.learn(&data[batch_index].0, &data[batch_index].1, i + 1, epochs);
        }
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize) -> () {
        println!("Starting the learnig process...");
        for i in 0..epochs {
            self.learn(&inputs, &targets, i + 1, epochs);
        }
    }

    pub fn learn(
        &mut self,
        inputs_batch: &Vec<Vec<f64>>,
        targets_batch: &Vec<Vec<f64>>,
        c_epoch: usize,
        epochs: usize,
    ) {
        init_progress_bar(inputs_batch.len());
        set_progress_bar_action(
            format!("{}/{}", c_epoch, epochs).as_str(),
            Color::Blue,
            Style::Bold,
        );
        for i in 0..inputs_batch.len() {
            self.update_all_gradients(&inputs_batch[i], &targets_batch[i]);
            inc_progress_bar();
        }

        set_progress_bar_action("Success", Color::Green, Style::Bold);

        for i in 0..self.layer_count {
            self.layers[i].apply_gradients(&(self.learning_rate / inputs_batch.len() as f64));
            self.layers[i].clear_gradients();
        }
        finalize_progress_bar();
    }

    pub fn update_all_gradients(&mut self, input: &Vec<f64>, target: &Vec<f64>) {
        self.feed_forwards(&input);

        // update the gradient of the output layer
        let mut node_values =
            self.layers[self.layer_count - 1].get_output_layer_node_value(&target);
        self.layers[self.layer_count - 1].update_gradients(&node_values);

        // update the gradient of the hidden layer
        for i in (0..self.layer_count - 2).rev() {
            let old_layer_weights = &self.clone().layers[i + 1].weights;
            let hidden_layer = &mut self.layers[i];
            node_values =
                hidden_layer.get_hidden_layer_node_value(&old_layer_weights, &node_values);
            self.layers[i].update_gradients(&node_values);
        }
    }

    #[allow(unused)]
    pub fn recognize(&mut self, input: &Vec<f64>) -> u8 {
        let output = self.feed_forwards(input);
        let mut max_index = 0;
        for i in 1..output.len() {
            if output[i] > output[max_index] {
                max_index = i;
            }
        }

        println!("{:?}", output);

        max_index as u8 + 1
    }

    pub fn test_accuracy(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
        let mut success_count = 0.0;

        init_progress_bar(inputs.len());

        for i in 0..inputs.len() {
            set_progress_bar_action(
                format!("acc: {}%", (success_count / (i + 1) as f64) * 100.0).as_str(),
                Color::Blue,
                Style::Bold,
            );
            let output = self.feed_forwards(&inputs[i]);
            let target = &targets[i];

            let mut max_index = 0;
            for i in 1..output.len() {
                if output[i] > output[max_index] {
                    max_index = i;
                }
            }

            if target[max_index] == 1.0 {
                success_count += 1.0;
            }

            inc_progress_bar();
        }

        let accuracy = success_count / inputs.len() as f64;

        set_progress_bar_action(
            format!("acc: {}%", accuracy * 100.0).as_str(),
            Color::Green,
            Style::Bold,
        );

        finalize_progress_bar();

        accuracy
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(&path)?;

        // layer count
        file.write(&(self.layer_count as u16 + 1).to_be_bytes())
            .expect("Can't write the layer_count to the saved file");

        // layers size
        file.write(&(self.layers[0].size_in as u16).to_be_bytes())
            .expect("Can't save the input layer");
        for i in 0..self.layer_count {
            file.write(&(self.layers[i].size_out as u16).to_be_bytes())
                .expect("Can't save layers");
        }

        for layer_index in 0..self.layer_count {
            for i in 0..self.layers[layer_index].weights.data.len() {
                for j in 0..self.layers[layer_index].weights.data[i].len() {
                    file.write(&(self.layers[layer_index].weights.data[i][j].to_be_bytes()))
                        .expect("Can't write weights to the save");
                }
            }
            for i in 0..self.layers[layer_index].biases.data.len() {
                file.write(&(self.layers[layer_index].biases.data[i][0].to_be_bytes()))
                    .expect("Can't write biases to the save");
            }
        }

        Ok(())
    }
}
