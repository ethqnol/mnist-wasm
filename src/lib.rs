use wasm_bindgen::prelude::*;

use nalgebra::DMatrix;
use std::f64::consts::E;
use serde_json;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub fn run_model(image : Vec<f64>) -> Vec<f64> {
    let model = Model::new();
    let result = model.query(image);
    let mut result_arr : Vec<f64> = Vec::new();
    for i in 0..result.nrows() {
        result_arr.push(result[(i, 0)]);
    }
    result_arr
}


#[derive(Serialize, Deserialize, Debug)]
pub struct ModelWeights {
    pub ih: Vec<f64>,
    pub ho: Vec<f64>,
}

pub struct Model {
    pub weights_input_hidden: DMatrix<f64>,
    pub weights_hidden_output: DMatrix<f64>,
}


impl Model {
    pub fn new() -> Self {
        let weights : ModelWeights = serde_json::from_str(include_str!("./model.json")).unwrap();
        Self {
            weights_input_hidden: DMatrix::from_vec(200, 784, weights.ih),
            weights_hidden_output: DMatrix::from_vec(10, 200, weights.ho),
        }
    }
    
    pub fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
    
    pub fn query(&self, input_nodes: Vec<f64>) -> DMatrix<f64> {
        let input: DMatrix<f64> = DMatrix::from_vec(784, 1, input_nodes);
    
        let hidden_inputs: DMatrix<f64> = self.weights_input_hidden.clone() * input;
        let hidden_outputs: DMatrix<f64> = hidden_inputs.map(|x| self.sigmoid(x));
    
        let final_inputs: DMatrix<f64> = self.weights_hidden_output.clone() * hidden_outputs;
        let final_outputs: DMatrix<f64> = final_inputs.map(|x| self.sigmoid(x));
        return final_outputs;
    }
}

