//author: Xuan Zhang
//version: 0.1.0
//date: 2025-03-22
//description: This file defines the LearningEngineDoubleQ class, which implements a simple solution based on Double Q-Learning.


#ifndef LEARNING_ENGINE_DOUBLE_Q_H
#define LEARNING_ENGINE_DOUBLE_Q_H

#include <torch/torch.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cassert>

// 定义 QNetwork 实现（两层全连接网络：输入2 -> 隐藏层64 -> 输出16）
struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QNetworkImpl(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) { 
        x = torch::nn::functional::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};
TORCH_MODULE(QNetwork);

// 前向声明 State 类，确保使用 const State* 时编译器知道它是个类型
class State;

// LearningEngineDoubleQ 类声明：实现基于 Double Q-Learning 的简单方案
class LearningEngineDoubleQ {
public:
    // 构造函数：传入学习率 alpha、折扣因子 gamma 和探索率 epsilon
    LearningEngineDoubleQ(float alpha, float gamma, float epsilon);

    // 根据当前状态选择动作（epsilon-greedy 策略），同时计算 max_to_avg_q_ratio
    uint32_t chooseAction(const State* state, float &max_to_avg_q_ratio);

    // 根据当前状态、动作、奖励和下一个状态更新网络参数
    void learn(const State* state, uint32_t action, int32_t reward, const State* next_state);

    // 同步目标网络参数（例如在初始化或每隔一定步数调用）
    void sync_target_network();

private:
    QNetwork online_qnetwork;
    QNetwork target_qnetwork;
    torch::optim::Adam optimizer;
    float gamma;    // 折扣因子
    float epsilon;  // 探索率
};

#endif // LEARNING_ENGINE_DOUBLE_Q_H
