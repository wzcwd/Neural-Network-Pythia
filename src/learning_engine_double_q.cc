// learning_engine_double_q.cc
// based on deep q learning
// wrote by xuan zhang

#include <torch/torch.h>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <ctime>
#include "learning_engine_double_q.h"  // State 类定义在此文件中
#include "scooby.h" // process_PC_delta 函数定义在此文件中
#include "scooby_helper.h"  // 确保包含 State 的完整定义


// 定义辅助函数 process_PC_delta，用于提取状态特征
// 如果在其它地方已有定义，可以使用它；否则这里提供一个实现
uint32_t process_PC_delta(uint64_t pc, int32_t delta) {
    const int DELTA_BITS = 7;
    uint32_t unsigned_delta = (delta < 0) ? ((-delta) + (1 << (DELTA_BITS - 1))) : delta;
    uint64_t tmp = pc;
    tmp = tmp << 7;
    tmp += unsigned_delta;
    // 这里简单返回 tmp 的低32位（或根据你的需求实现哈希函数）
    return static_cast<uint32_t>(tmp);
}

// LearningEngineDoubleQ 构造函数
LearningEngineDoubleQ::LearningEngineDoubleQ(float alpha, float gamma, float epsilon)
    : online_qnetwork(QNetwork(2, 64, 16)),
      target_qnetwork(QNetwork(2, 64, 16)),
      optimizer(online_qnetwork->parameters(), torch::optim::AdamOptions(alpha)),
      gamma(gamma),
      epsilon(epsilon)
{
    srand(static_cast<unsigned>(time(NULL)));
    sync_target_network();
}

// chooseAction: 根据当前状态采用 ε-greedy 策略选择动作，同时计算 max_to_avg_q_ratio
uint32_t LearningEngineDoubleQ::chooseAction(const State* state, float &max_to_avg_q_ratio) {
    float feature1 = static_cast<float>(state->local_delta_sig2);
    float feature2 = static_cast<float>(process_PC_delta(state->pc, state->delta));
    torch::Tensor input = torch::tensor({feature1, feature2}).unsqueeze(0);  // shape: [1,2]
    
    torch::Tensor q_values = online_qnetwork->forward(input); // 在线网络输出，shape: [1,16]
    auto q_vec = q_values.squeeze(0).to(torch::kCPU);
    
    uint32_t action = 0;
    float rand_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (rand_val < epsilon) {
        action = rand() % 16;
    } else {
        // 使用元组解构获取最大值和对应的索引
        auto result = q_vec.max(0);  // result 为 tuple (max_vals, max_idx)
        auto max_vals = std::get<0>(result);
        auto max_idx  = std::get<1>(result);
        action = max_idx.item<int>();
    }
    float max_q = q_vec.max().item<float>();
    float avg_q = q_vec.mean().item<float>();
    max_to_avg_q_ratio = (avg_q > 0.0f) ? (max_q / avg_q) : 1.0f;
    return action;
}

// learn: 根据当前状态、动作、奖励和下一个状态更新网络参数（Double Q-Learning 更新公式）
void LearningEngineDoubleQ::learn(const State* state, uint32_t action, int32_t reward, const State* next_state) {
    float feature1 = static_cast<float>(state->local_delta_sig2);
    float feature2 = static_cast<float>(process_PC_delta(state->pc, state->delta));
    torch::Tensor state_tensor = torch::tensor({feature1, feature2}).unsqueeze(0);
    
    float next_feature1 = static_cast<float>(next_state->local_delta_sig2);
    float next_feature2 = static_cast<float>(process_PC_delta(next_state->pc, next_state->delta));
    torch::Tensor next_state_tensor = torch::tensor({next_feature1, next_feature2}).unsqueeze(0);
    
    // 在线网络选择下一个状态的最佳动作
    torch::Tensor next_q_online = online_qnetwork->forward(next_state_tensor);
    int next_action = next_q_online.argmax(1).item<int>();
    
    // 使用目标网络计算下一个状态下该动作对应的 Q 值
    torch::Tensor next_q_target = target_qnetwork->forward(next_state_tensor);
    float next_q = next_q_target[0][next_action].item<float>();
    
    // 计算目标值
    float target_value = reward + gamma * next_q;
    
    // 在线网络计算当前状态下所有动作的 Q 值
    torch::Tensor q_values = online_qnetwork->forward(state_tensor);
    torch::Tensor current_q = q_values[0][action];
    torch::Tensor target = torch::tensor(target_value);
    
    torch::Tensor loss = torch::mse_loss(current_q, target);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}

// sync_target_network: 将在线网络参数复制到目标网络中（定期调用）
void LearningEngineDoubleQ::sync_target_network() {
    torch::NoGradGuard no_grad;
    auto online_params = online_qnetwork->named_parameters();
    auto target_params = target_qnetwork->named_parameters();
    for (auto& item : online_params) {
        auto key = item.key();
        target_params[key].copy_(item.value());
    }
}
