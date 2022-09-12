use crate::neural_network;
use macroquad::prelude::*;

pub async fn run_visualization (neural_net: neural_network::NeuralNetwork) {
    let mut layers = vec![0; neural_net.layers.len() + 1];
    
    let mut i = 0;
    while i < neural_net.layers.len() {
        if i == 0 {
            layers[0] = neural_net.layers[i].num_nodes_in;
        }

        layers[i + 1] = neural_net.layers[i].num_nodes_out;
        
        i += 1;
    }

    fn draw_layer(width: f32,  height: f32, node_count: i32, layer_count: i32, layer_number: i32, max_layer_height: i32) {
        let nn_top_edge = (screen_height() - height) / 2.0;
        let nn_left_edge = (screen_width() - width) / 2.0;

        let distance_from_left = if layer_number == 0 { 0.0 } else { width / (layer_count - 1) as f32 * layer_number as f32 };

        for i in 0..node_count {
            let step = height / (max_layer_height - 1) as f32 / 2.0;
            let starting_distance_from_top = (max_layer_height - node_count) as f32 * step;
            let distance_from_top = starting_distance_from_top + (step as f32 * 2.0 * i as f32);

            draw_circle(nn_left_edge + distance_from_left, nn_top_edge + distance_from_top, 10.0, BLACK);
        }
    }

    fn draw_nn_background(width: f32, height: f32) {
        let nn_left_edge = (screen_width() - width) / 2.0;
        let nn_top_edge = (screen_height() - height) / 2.0;

        draw_rectangle(nn_left_edge, nn_top_edge, width, height, GRAY);
    }

    loop {
        clear_background(WHITE);

        let nn_width = screen_width() * 0.8;
        let nn_height = screen_height() * 0.4;

        let max_layer_height = layers
            .iter()
            .max()
            .unwrap();

        for (i, node_count) in layers.iter().enumerate() {
            draw_layer(nn_width, nn_height, *node_count as i32, layers.len() as i32, i as i32, *max_layer_height as i32);
        }

        next_frame().await
    }
}