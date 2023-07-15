use std::env;
use std::fs::File;
use std::path::Path;

use crate::lib::training_data::TrainingData;

pub mod lib;

fn main() {
    let args: Vec<String> = env::args().collect();

    // cargo run image_path
    if args.len() == 2 {
        // run the ai on the selected path
        println!("Oppening the image...")
    } else if args.len() == 3 && args[1] == "train" {
        println!("Oppening the training set...");

        // open the training file
        let mut f = File::open(&args[2]).expect("The file provided is invalid or unreachable");

        println!("Getting the dataset meta...");
        let training_data = TrainingData::new(&mut f);

        println!("Extracting the images...");
        for (count, image) in training_data.take(20).enumerate() {
            let name = count.to_string();
            image
                .save(Path::new(&(format!("training_set/{}.png", name))))
                .expect("Can't save the image");
        }
    }
}
