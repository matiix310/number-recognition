use std::{fs::File, io::Read};

pub struct TrainingData {
    // images
    pub image_magic_number: u32,
    pub image_count: u32,
    pub rows_count: u32,
    pub cols_count: u32,
    file_images: File,

    // labels
    pub label_magic_number: u32,
    pub label_count: u32,
    file_labels: File,
}

impl TrainingData {
    pub fn new(file_images: &mut File, file_labels: &mut File) -> TrainingData {
        let mut image_magic_number = [0; 4];
        let mut label_magic_number = [0; 4];
        let mut image_count = [0; 4];
        let mut label_count = [0; 4];
        let mut rows_count = [0; 4];
        let mut cols_count = [0; 4];

        file_images
            .read(&mut image_magic_number)
            .expect("Can't read the images magic number");
        file_images
            .read(&mut image_count)
            .expect("Can't read the image count");
        file_images
            .read(&mut rows_count)
            .expect("Can't read the rows count");
        file_images
            .read(&mut cols_count)
            .expect("Can't read the cols count");

        file_labels
            .read(&mut label_magic_number)
            .expect("Can't read the labels magic number");
        file_labels
            .read(&mut label_count)
            .expect("Can't read the label count");

        TrainingData {
            image_magic_number: Self::bytes_to_u32(image_magic_number),
            image_count: Self::bytes_to_u32(image_count),
            rows_count: Self::bytes_to_u32(rows_count),
            cols_count: Self::bytes_to_u32(cols_count),
            file_images: file_images
                .try_clone()
                .expect("Can't clone the training data (->images) to store them"),
            label_magic_number: Self::bytes_to_u32(label_magic_number),
            label_count: Self::bytes_to_u32(label_count),
            file_labels: file_labels
                .try_clone()
                .expect("Can't clone the training data (->labels) to store them"),
        }
    }

    pub fn bytes_to_u32(bytes: [u8; 4]) -> u32 {
        ((bytes[3] as u32) << 0)
            + ((bytes[2] as u32) << 8)
            + ((bytes[1] as u32) << 16)
            + ((bytes[0] as u32) << 24)
    }
}

impl Iterator for TrainingData {
    type Item = (Vec<f64>, Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        // read the label
        let mut label_buffer = [0; 1];
        let mut outputs: [f64; 10] = [0.0; 10];

        self.file_labels
            .read(&mut label_buffer)
            .expect("Can't read the label");

        outputs[label_buffer[0] as usize] = 1.0;

        // read the image pixels
        let mut inputs: Vec<f64> = Vec::<f64>::new();
        for _ in 0..self.rows_count {
            for _ in 0..self.cols_count {
                let mut pixel = [0; 1];
                self.file_images
                    .read(&mut pixel)
                    .expect("Can't read the pixel");
                inputs.push(pixel[0] as f64)
            }
        }

        let image: Self::Item = (inputs, outputs.to_vec());
        Option::Some(image)
    }
}

impl Clone for TrainingData {
    fn clone(&self) -> Self {
        Self {
            image_magic_number: self.image_magic_number.clone(),
            image_count: self.image_count.clone(),
            rows_count: self.rows_count.clone(),
            cols_count: self.cols_count.clone(),
            file_images: self
                .file_images
                .try_clone()
                .expect("Can't clone the training data (->images) to store them"),
            label_magic_number: self.label_magic_number.clone(),
            label_count: self.label_count.clone(),
            file_labels: self
                .file_labels
                .try_clone()
                .expect("Can't clone the training data (->labels) to store them"),
        }
    }
}
