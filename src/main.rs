use std::env;
use std::fs::File;
use std::io::{stdin, stdout, Write};

use crate::network::activations;
use crate::network::network::Network;
use crate::network::training_data::TrainingData;

mod matrix;
mod network;

fn main() {
    let args: Vec<String> = env::args().collect();

    // cargo run image_path
    if args.len() == 2 {
        // run the ai on the selected path
        println!("Oppening the image...")
    } else if args.len() == 4 && args[1] == "train" {
        println!("Oppening the training set...");

        let (input_model_path, output_model_path) = ask_model_in_out();

        // open the image training file
        let mut f_images =
            File::open(&args[2]).expect("The file provided is invalid or unreachable");

        // open the label training file
        let mut f_labels =
            File::open(&args[3]).expect("The file provided is invalid or unreachable");

        println!("Getting the dataset meta...");
        let training_data = TrainingData::new(&mut f_images, &mut f_labels);

        println!(
            "Images:\n  Magic number: {} | Count: {} | Size: {}x{}",
            training_data.image_magic_number,
            training_data.image_count,
            training_data.cols_count,
            training_data.rows_count,
        );
        println!(
            "Labels:\n  Magic number: {} | Count: {}",
            training_data.label_magic_number, training_data.label_count,
        );

        println!("Loading the training data...");
        let mut inputs = Vec::<Vec<f64>>::new();
        let mut targets = Vec::<Vec<f64>>::new();

        for (input, target) in training_data.clone().into_iter().take(100000) {
            inputs.push(input);
            targets.push(target);
        }

        let layers_struct = vec![
            (training_data.rows_count * training_data.cols_count) as usize,
            100,
            10,
        ];

        let learning_rate = 1.0;
        let activation_function = activations::SIGMOID;

        let mut network = if input_model_path == "" {
            Network::new(&layers_struct, &learning_rate, activation_function)
        } else {
            println!("Loading model from: {}...", input_model_path);
            Network::load_from_file(&input_model_path, &learning_rate, activation_function)
        };

        network.train_with_batch(&inputs, &targets, 10000, 5); // -> 58% // -> 70%
                                                               // network.train(&inputs, &targets, 10);

        network.test_accuracy(&inputs, &targets);

        if output_model_path != "" {
            network.save(&output_model_path).ok();
            println!("File saved at location: {}", output_model_path);
        }
    } else {
        println!(
            "Invalid command : \n  cargo run train path/to/images_dataset path/to/labels_dataset\n  cargo run path/to/image"
        )
    }

    fn ask_question(question: &str) -> String {
        let mut s = String::new();
        println!("{}", question);
        stdin()
            .read_line(&mut s)
            .expect("Did not enter a correct string");
        if let Some('\n') = s.chars().next_back() {
            s.pop();
        }
        if let Some('\r') = s.chars().next_back() {
            s.pop();
        };

        s
    }

    fn ask_model_in_out() -> (String, String) {
        // let _ = stdout().flush();

        let input_path = ask_question("Input model path (enter if new one): ");
        // println!("input: '{}'", input_path);
        let output_path = ask_question("Output model path (enter if no output): ");
        // println!("output: '{}'", output_path);

        (input_path, output_path)
    }
}
