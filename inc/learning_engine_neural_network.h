//
// Created by wzc on 3/21/25.
//
#ifndef LEARNING_ENGINE_NEURAL_NETWORK_H
#define LEARNING_ENGINE_NEURAL_NETWORK_H

#include <vector>
#include <cstdint> // for uint32_t, int32_t
#include <torch/torch.h>

struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QNetworkImpl(int input_dim, int hidden_dim, int output_dim);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(QNetwork);

class State;


class LearningEngineNeuralNetwork {
public:
    // Constructor
    LearningEngineNeuralNetwork(float alpha, float gamma, float epsilon);

    // Select action
    uint32_t chooseAction(const State* state, float &max_to_avg_q_ratio);

    // Train / Learn
    void learn(const State* state, uint32_t action, int32_t reward, const State* next_state);

private:
    QNetwork qnetwork;              // The defined q-network
    torch::optim::Adam optimizer;   // Optimizer: Adjust the parameters (weights and biases) of the neural network based on the result of the loss function to make the model perform better.
    float gamma;    // Discount factor
    float alpha;
    float epsilon;  // Exploration rate in ε-greedy strategy
};



#endif //LEARNING_ENGINE_NEURAL_NETWORK_H
