mod neural_network;

#[macroquad::main("Neural Network Visualization")]
async fn main() {
    let my_neural_net = neural_network::NeuralNetwork::new(vec![2, 3, 2]);
    neural_network::visualization::run_visualization(my_neural_net).await;
}