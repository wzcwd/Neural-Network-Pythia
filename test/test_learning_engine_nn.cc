// This file is used to test the learning_engine_neural_network
#include "../inc/learning_engine_neural_network.h"
#include <iostream>
#include <cassert>

// // Simulated State struct
struct State {
    uint64_t pc;
    int32_t delta;
    uint32_t local_delta_sig2;
};

// // Simulate a test case
void test_learning_engine_nn() {
    std::cout << "[Test] Starting neural network engine test..." << std::endl;

    // initialize the engine
    LearningEngineNeuralNetwork engine(0.001f, 0.95f, 0.1f);  // alpha, gamma, epsilon

    // Construct test states
    State state1 = {0x12345, 5, 17};
    State state2 = {0x12355, 6, 21};
    float max_to_avg_q_ratio = 0.0;

    // Test chooseAction()
    uint32_t action = engine.chooseAction(&state1, max_to_avg_q_ratio);
    std::cout << "[Test] Action selected: " << action << std::endl;
    assert(action < 16);

    // Test learn()
    int32_t reward = 20;
    engine.learn(&state1, action, reward, &state2);

    std::cout << "[Test] Neural network learning step completed successfully." << std::endl;
}

// run the test
int main() {
    test_learning_engine_nn();
    return 0;
}