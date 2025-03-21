#include <torch/torch.h>
#include <vector>
#include <cstdlib>  // for rand()
#include <cassert>
#include <stdint.h>
#include "learning_engine_neural_network.h"


using namespace std;

// define constant, subject to change based on demand（
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

// Define Q-network （single hidden layer：2-neuron input-> 64-neuron hidden layer -> 16-neuron output）
struct QNetworkImpl : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QNetworkImpl(int input_dim, int hidden_dim, int output_dim)
    {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::nn::functional::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }


};
TORCH_MODULE(QNetwork);

// NeuralNetworkEngine 类实现
class LearningEngineNeuralNetwork {
public:
    // 构造函数，传入学习率（alpha）、折扣因子（gamma）和探索率（epsilon）
    LearningEngineNeuralNetwork(float alpha, float gamma, float epsilon)
      : qnetwork(QNetwork(2, 64, 16)),
        optimizer(qnetwork->parameters(), torch::optim::AdamOptions(alpha)),
        gamma(gamma),
        epsilon(epsilon)
    {
        // 初始化时可设置随机种子
        srand((unsigned)time(NULL));
    }

    // 根据当前状态选择动作，采用 epsilon-贪心策略
    uint32_t chooseAction(const State* state, float &max_to_avg_q_ratio)
    {
        // 提取两个输入特征：
        // 特征1：state->local_delta_sig2
        // 特征2：pc+delta，通过 process_PC_delta 计算
        float feature1 = static_cast<float>(state->local_delta_sig2);
        float feature2 = static_cast<float>(process_PC_delta(0, state->pc, state->delta));

        torch::Tensor input = torch::tensor({feature1, feature2}).unsqueeze(0); // shape [1,2]
        torch::Tensor q_values = qnetwork->forward(input); // 输出 shape [1,16]
        // 将输出转换到 CPU
        auto q_vec = q_values.squeeze(0).to(torch::kCPU);

        uint32_t action = 0;
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        if(rand_val < epsilon) {
            // 随机选取一个动作
            action = rand() % 16;
        } else {
            // 否则选择 Q 值最大的动作
            auto max_result = q_vec.max(0);
            action = max_result.indices.item<int>();
        }
        // 计算 max_to_avg_q_ratio（用于后续可能的动态调整）
        float max_q = q_vec.max().item<float>();
        float avg_q = q_vec.mean().item<float>();
        max_to_avg_q_ratio = (avg_q > 0) ? max_q / avg_q : 1.0;
        return action;
    }

    // 根据当前状态、动作、奖励和下一个状态更新网络参数（基于 DQN 的更新公式）
    void learn(const State* state, uint32_t action, int32_t reward, const State* next_state)
    {
        // 当前状态的特征向量
        float feature1 = static_cast<float>(state->local_delta_sig2);
        float feature2 = static_cast<float>(process_PC_delta(0, state->pc, state->delta));
        torch::Tensor state_tensor = torch::tensor({feature1, feature2}).unsqueeze(0); // shape [1,2]

        // 下一个状态的特征向量
        float next_feature1 = static_cast<float>(next_state->local_delta_sig2);
        float next_feature2 = static_cast<float>(process_PC_delta(0, next_state->pc, next_state->delta));
        torch::Tensor next_state_tensor = torch::tensor({next_feature1, next_feature2}).unsqueeze(0);

        // 当前状态下所有动作的 Q 值
        torch::Tensor q_values = qnetwork->forward(state_tensor); // shape [1,16]
        // 下一个状态的 Q 值
        torch::Tensor next_q_values = qnetwork->forward(next_state_tensor); // shape [1,16]

        // DQN 的目标值：reward + gamma * max_a Q(next_state, a)
        float next_max = next_q_values.max(1).values.item<float>();
        float target_value = reward + gamma * next_max;

        // 取出当前动作对应的 Q 值
        torch::Tensor current_q = q_values[0][action];
        torch::Tensor target = torch::tensor(target_value);

        // 均方误差损失
        torch::Tensor loss = torch::mse_loss(current_q, target);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

private:
    QNetwork qnetwork;
    torch::optim::Adam optimizer;
    float gamma;    // 折扣因子
    float epsilon;  // ε-贪心策略中的探索率
};