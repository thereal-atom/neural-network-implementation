use utils::math::sigmoid;

struct Layer {
    num_nodes_in: usize,
    num_nodes_out: usize,
    cost_gradient_w: Vec<Vec<f32>>,
    cost_gradient_b: Vec<f32>,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {
    pub fn new(num_nodes_in: usize, num_nodes_out: usize) -> Self {
        Self {
            num_nodes_in,
            num_nodes_out,
            cost_gradient_w: vec![vec![0.0; num_nodes_out]; num_nodes_in],
            weights: vec![vec![0.0; num_nodes_out]; num_nodes_in],
            cost_gradient_b: vec![0.0; num_nodes_out],
            biases: vec![0.0; num_nodes_out],
        }
    }

    pub fn apply_gradients(&mut self, learn_rate: f32) {
        let mut node_out = 0;
        while node_out < self.num_nodes_out {
            self.biases[node_out] -= self.cost_gradient_b[node_out] * learn_rate;

            let mut node_in = 0;
            while node_in < self.num_nodes_in {
                self.weights[node_in][node_out] = self.cost_gradient_w[node_in][node_out] * learn_rate;

                node_in += 1;
            }

            node_out += 1;
        }
    }

    fn activation_function(&self, weighted_input: f32) -> f32 {
        let activation = if weighted_input > 0.0 { 1.0 }
        else { 0.0 };

        sigmoid(activation)
    }

    fn node_cost(&self, output_activation: f32, expected_output: f32) -> f32 {
        let error = output_activation - expected_output;
        error * error
    }

    pub fn calculate_outputs(&self, inputs: Vec<f32>) -> Vec<f32> {
        // if inputs.len() != self.num_nodes_in { println!("There should only be {} inputs", self.num_nodes_in) };
        
        let mut activations: Vec<f32> = vec![0.0; self.num_nodes_out];

        let mut node_out: usize = 0;
        while node_out < self.num_nodes_out {
            let mut weighted_input: f32 = self.biases[node_out];

            let mut node_in: usize = 0;
            while node_in < self.num_nodes_in {
                weighted_input += inputs[node_in] * self.weights[node_in][node_out];

                node_in += 1;
            }

            activations[node_out] = self.activation_function(weighted_input);

            node_out += 1;
        }

        activations
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
    input_value: f32,
}

struct DataPoint {
    inputs: Vec<f32>,
    expected_outputs: Vec<f32>
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers_list: Vec<Layer> = Vec::new();

        let mut i = 0;
        while i < layer_sizes.len() - 1 {
            layers_list.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));

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

fn main() {
    let my_neural_net = NeuralNetwork::new(vec![2, 3, 2]);
    let result = my_neural_net.classify(vec![0.0, 0.0]);

    println!("Result: {}", result);
}