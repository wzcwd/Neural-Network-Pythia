#include "../inc/learning_engine_neural_network.h"
#include <iostream>
#include <cassert>

// 模拟 State 类（你在工程中可能是更完整的定义）
struct State {
    uint64_t pc;
    int32_t delta;
    uint32_t local_delta_sig2;
};

// 模拟一个测试用例
void test_learning_engine_nn() {
    std::cout << "[Test] Starting neural network engine test..." << std::endl;

    // 初始化引擎
    LearningEngineNeuralNetwork engine(0.001f, 0.95f, 0.1f);  // alpha, gamma, epsilon

    // 构造测试状态
    State state1 = {0x12345, 5, 17};
    State state2 = {0x12355, 6, 21};
    float max_to_avg_q_ratio = 0.0;

    // 测试 chooseAction()
    uint32_t action = engine.chooseAction(&state1, max_to_avg_q_ratio);
    std::cout << "[Test] Action selected: " << action << std::endl;
    assert(action < 16);

    // 测试 learn()
    int32_t reward = 20;
    engine.learn(&state1, action, reward, &state2);

    std::cout << "[Test] Neural network learning step completed successfully." << std::endl;
}

// 加上 main 函数以运行测试
int main() {
    test_learning_engine_nn();
    return 0;
}