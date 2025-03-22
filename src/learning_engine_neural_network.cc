#include "learning_engine_neural_network.h"
#include "scooby.h"

#include <torch/torch.h>
#include <vector>
#include <cstdlib>  // for rand()
#include <cassert>
#include <stdint.h>

using namespace std;

// define constant, subject to change based on demand
#define DELTA_BITS 7

//  Jenkins hash function(get from original approach)
uint32_t jenkins(uint32_t key)
{
    key += (key << 12);
    key ^= (key >> 22);
    key += (key << 4);
    key ^= (key >> 9);
    key += (key << 10);
    key ^= (key >> 2);
    key += (key << 7);
    key ^= (key >> 12);
    return key;
}

//  folded_xor hash function(get from original approach)
uint32_t folded_xor(uint64_t value, uint32_t num_folds)
{
    assert(num_folds > 1);
    assert((num_folds & (num_folds-1)) == 0);
    uint32_t mask = 0;
    uint32_t bits_in_fold = 64 / num_folds;
    if (num_folds == 2)
    {
        mask = 0xffffffff;
    }
    else
    {
        mask = (1ul << bits_in_fold) - 1;
    }
    uint32_t folded_value = 0;
    for (uint32_t fold = 0; fold < num_folds; ++fold)
    {
        folded_value ^= ((value >> (fold * bits_in_fold)) & mask);
    }
    return folded_value;
}


// pc + delta:  get from original approach
uint32_t process_PC_delta(uint64_t pc, int32_t delta)
{
    uint32_t unsigned_delta = (delta < 0) ? ((-delta) + (1 << (DELTA_BITS - 1))) : delta;
    uint64_t tmp = pc;
    tmp = tmp << 7;
    tmp += unsigned_delta;
    uint32_t raw_index = folded_xor(tmp, 2);
    uint32_t hashed_index = jenkins(raw_index);
    return (hashed_index);
}

// Q-Network implementation
QNetworkImpl::QNetworkImpl(int input_dim, int hidden_dim, int output_dim)
{
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
}

torch::Tensor QNetworkImpl::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return x;
}

// LearningEngineNeuralNetwork constructor
LearningEngineNeuralNetwork::LearningEngineNeuralNetwork(float alpha, float gamma, float epsilon)
    : qnetwork(QNetwork(2, 64, 16)),
      optimizer(qnetwork->parameters(), torch::optim::AdamOptions(alpha)),
      gamma(gamma), alpha(alpha), epsilon(epsilon)
{
    srand((unsigned)time(NULL));
}

// Action selection (ε-greedy strategy)
uint32_t LearningEngineNeuralNetwork::chooseAction(const State* state, float &max_to_avg_q_ratio)
{
    float f1 = static_cast<float>(state->local_delta_sig2);
    float f2 = static_cast<float>(process_PC_delta(state->pc, state->delta));
    torch::Tensor input = torch::tensor({f1, f2}).unsqueeze(0); // shape: [1,2]

    torch::Tensor q_values = qnetwork->forward(input);
    auto q_vec = q_values.squeeze(0).to(torch::kCPU);

    uint32_t action = 0;
    float rand_val = static_cast<float>(rand()) / RAND_MAX;

    if (rand_val < epsilon) {
        action = rand() % 16;
    } else {
        action = q_vec.argmax().item<int>();
    }

    float max_q = q_vec.max().item<float>();
    float avg_q = q_vec.mean().item<float>();
    max_to_avg_q_ratio = (avg_q > 0.0f) ? (max_q / avg_q) : 1.0f;

    return action;
}



// Network update (DQN style)
void LearningEngineNeuralNetwork::learn(const State* state, uint32_t action, int32_t reward, const State* next_state)
{
    float f1 = static_cast<float>(state->local_delta_sig2);
    float f2 = static_cast<float>(process_PC_delta(state->pc, state->delta));
    torch::Tensor state_tensor = torch::tensor({f1, f2}).unsqueeze(0);

    float next_f1 = static_cast<float>(next_state->local_delta_sig2);
    float next_f2 = static_cast<float>(process_PC_delta(next_state->pc, next_state->delta));
    torch::Tensor next_tensor = torch::tensor({next_f1, next_f2}).unsqueeze(0);

    torch::Tensor q_values = qnetwork->forward(state_tensor);
    torch::Tensor next_q_values = qnetwork->forward(next_tensor);
    auto max_result = next_q_values.max(1);
    float next_max = std::get<0>(max_result).item<float>();
    float target_value = reward + gamma * next_max;

    torch::Tensor current_q = q_values[0][action];
    torch::Tensor target = torch::tensor(target_value);

    torch::Tensor loss = torch::mse_loss(current_q, target);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}




