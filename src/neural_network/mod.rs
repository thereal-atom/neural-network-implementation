mod layer;
pub mod visualization;

pub struct NeuralNetwork {
    pub layers: Vec<layer::Layer>,
    input_value: f32,
}

struct DataPoint {
    inputs: Vec<f32>,
    expected_outputs: Vec<f32>
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers_list: Vec<layer::Layer> = Vec::new();

        let mut i = 0;
        while i < layer_sizes.len() - 1 {
            layers_list.push(layer::Layer::new(layer_sizes[i], layer_sizes[i + 1]));

            i += 1;
        };

        Self {
            layers: layers_list,
            input_value: 0.0,
        }
    }

    fn calculate_outputs(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.calculate_outputs(inputs);
        };

        inputs
    }

    fn cost(&self, data_point: DataPoint) -> f32 {
        let outputs = self.calculate_outputs(data_point.inputs);
        let output_layer = &self.layers[self.layers.len() - 1];
        let mut cost = 0.0;

        let mut node_out = 0;
        while node_out < outputs.len() {
            cost += output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out]);

            node_out += 1;
        }

        cost
    }

    pub fn init(&mut self, random_value: f32) {
        self.input_value = random_value;
    }

    pub fn learn(&mut self, learn_rate: f32) {
        let h = 0.00001;
        let delta_output = self.function(self.input_value + h) - self.function(self.input_value);
        let slope = delta_output / h;

        self.input_value -= slope * learn_rate;
    }

    fn function(&self, x: f32) -> f32 {
        0.2 * x.powi(4) + 0.1 * x.powi(3) - x.powi(2) + 2.0
    }

    pub fn classify(&self, inputs: Vec<f32>) -> f32 {
        let outputs = self.calculate_outputs(inputs);
        // *outputs
        //     .iter()
        //     .max()
        //     .unwrap()

        outputs[0]
    }
}