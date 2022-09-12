use utils::math::sigmoid;

pub struct Layer {
    pub num_nodes_in: usize,
    pub num_nodes_out: usize,
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

    pub fn node_cost(&self, output_activation: f32, expected_output: f32) -> f32 {
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