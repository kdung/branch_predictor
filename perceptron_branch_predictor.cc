#include "simulator.h"
#include "perceptron_branch_predictor.h"

PerceptronBranchPredictor::PerceptronBranchPredictor(String name,
        core_id_t core_id) :
        BranchPredictor(name, core_id),
        N(1 << PERCEPTRON_INDEX_BIT),
        theta(1.93 * HISTORY_LEN + 14),
        perceptron_table(N, std::vector<SInt32>(HISTORY_LEN + 1, 0)),
        history_reg(0xFFFFFFFF),
        last_y_predict(1) {
}

PerceptronBranchPredictor::~PerceptronBranchPredictor() {
}

bool PerceptronBranchPredictor::predict(IntPtr ip, IntPtr target) {
    std::vector<SInt32> perceptron = get_perceptron(ip);
    last_y_predict = BIAS_WEIGHT * perceptron[0];
    for (UInt32 i = 0; i < HISTORY_LEN; i++) {
        int xi = history_reg >> i;
        if (xi % 2) {
            last_y_predict += perceptron[i + 1];
        } else {
            last_y_predict -= perceptron[i + 1];
        }
    }
    return last_y_predict > 0;
}

void PerceptronBranchPredictor::update(bool predicted, bool actual, IntPtr ip,
        IntPtr target) {
    updateCounters(predicted, actual);
    std::vector<SInt32> perceptron = get_perceptron(ip);
    if (predicted != actual || abs(last_y_predict) <= theta) {
        perceptron[0] =
                actual ?
                        increase_weight(perceptron[0]) :
                        decrease_weight(perceptron[0]);
        for (UInt32 i = 0; i < HISTORY_LEN; i++) {
            int xi = history_reg >> i;
            if (xi % 2 == actual) {
                perceptron[i+1] = increase_weight(perceptron[i+1]);
            } else {
                perceptron[i+1] = decrease_weight(perceptron[i+1]);
            }
        }
    }
    history_reg = (history_reg << 1);
    if (actual) {
        history_reg++;
    }
}

std::vector<SInt32> PerceptronBranchPredictor::get_perceptron(IntPtr ip) {
    UInt32 index = (ip >> 4) % N;
    return perceptron_table[index];
}

SInt32 PerceptronBranchPredictor::increase_weight(SInt32 weight) {
    return (weight < ((1 << WEIGHT_BIT_SIZE) -1)) ? weight + 1 : weight;
}

SInt32 PerceptronBranchPredictor::decrease_weight(SInt32 weight) {
    return (weight > -(1 << WEIGHT_BIT_SIZE)) ? weight - 1 : weight;
}
