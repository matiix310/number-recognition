use std::{fs::File, io::Read};

use image::{ImageBuffer, Luma};

pub struct TrainingData {
    #[allow(dead_code)]
    magic_number: u32,
    #[allow(dead_code)]
    image_count: u32,
    rows_count: u32,
    cols_count: u32,
    file: File,
}

impl TrainingData {
    pub fn new(file: &mut File) -> TrainingData {
        let mut magic_number = [0; 4];
        let mut image_count = [0; 4];
        let mut rows_count = [0; 4];
        let mut cols_count = [0; 4];

        file.read(&mut magic_number)
            .expect("Can't read the magic number");
        file.read(&mut image_count)
            .expect("Can't read the image count");
        file.read(&mut rows_count)
            .expect("Can't read the rows count");
        file.read(&mut cols_count)
            .expect("Can't read the cols count");

        TrainingData {
            magic_number: Self::bytes_to_u32(magic_number),
            image_count: Self::bytes_to_u32(image_count),
            rows_count: Self::bytes_to_u32(rows_count),
            cols_count: Self::bytes_to_u32(cols_count),
            file: file
                .try_clone()
                .expect("Can't clone the training data to store them"),
        }
    }

    pub fn bytes_to_u32(bytes: [u8; 4]) -> u32 {
        ((bytes[3] as u32) << 0)
            + ((bytes[2] as u32) << 8)
            + ((bytes[1] as u32) << 16)
            + ((bytes[0] as u32) << 24)
    }

    pub fn get_width(&mut self) -> u32 {
        self.cols_count
    }

    pub fn get_height(&mut self) -> u32 {
        self.rows_count
    }
}

impl Iterator for TrainingData {
    type Item = ImageBuffer<Luma<u8>, Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut image: Self::Item = ImageBuffer::new(self.cols_count, self.rows_count);

        for y in 0..self.rows_count {
            for x in 0..self.cols_count {
                let mut pixel = [0; 1];
                self.file.read(&mut pixel).expect("Can't read the pixel");
                image.put_pixel(x, y, Luma::<u8>::from(pixel))
            }
        }

        Option::Some(image)
    }
}
