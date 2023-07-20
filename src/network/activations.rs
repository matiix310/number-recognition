use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| (SIGMOID.function)(x) * (1.0 - (SIGMOID.function)(x)),
};

#[allow(unused)]
pub const RELU: Activation = Activation {
    function: &|x| if x > 0.0 { x } else { 0.01 * x },
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.01 },
};
