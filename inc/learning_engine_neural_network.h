//
// Created by wzc on 3/21/25.
//
#ifndef LEARNING_ENGINE_NEURAL_NETWORK_H
#define LEARNING_ENGINE_NEURAL_NETWORK_H

#include <vector>
#include <cstdint> // for uint32_t, int32_t
#include <torch/torch.h>



// 如果 "State" 是另一个类/结构体，在别的头文件声明，
// 但这里只用指针，就可用前向声明:
class State;

// QNetwork 在上面的 .cc 里是由 TORCH_MODULE(QNetwork) 定义的；
// 它来自 <torch/torch.h> 中的宏，因此这里包含 <torch/torch.h> 即可。

class LearningEngineNeuralNetwork {
public:
    // 构造函数
    LearningEngineNeuralNetwork(float alpha, float gamma, float epsilon);

    // 选择动作
    uint32_t chooseAction(const State* state, float &max_to_avg_q_ratio);

    // 训练/学习
    void learn(const State* state, uint32_t action, int32_t reward, const State* next_state);

private:
    QNetwork qnetwork;              // 你定义的神经网络 (2->64->16)
    torch::optim::Adam optimizer;   // 优化器
    float gamma;    // 折扣因子
    float alpha;
    float epsilon;  // ε-贪心策略中的探索率
};



#endif //LEARNING_ENGINE_NEURAL_NETWORK_H
