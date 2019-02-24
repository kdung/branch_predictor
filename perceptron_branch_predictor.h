#ifndef PERCEPTRON_BRANCH_PREDICTOR_H
#define PERCEPTRON_BRANCH_PREDICTOR_H

#include "branch_predictor.h"
#include <vector>

class PerceptronBranchPredictor: public BranchPredictor {
public:
    PerceptronBranchPredictor(String name, core_id_t core_id);
    ~PerceptronBranchPredictor();
    bool predict(IntPtr ip, IntPtr target);
    void update(bool predicted, bool actual, IntPtr ip, IntPtr target);

private:
    static const UInt32 PERCEPTRON_INDEX_BIT = 11;
    static const UInt32 BIAS_WEIGHT = 1;
    static const UInt32 HISTORY_LEN = 54;
    static const UInt32 WEIGHT_BIT_SIZE = 7;
    std::vector<SInt32> get_perceptron(IntPtr ip);
    SInt32 increase_weight(SInt32 weight);
    SInt32 decrease_weight(SInt32 weight);
    UInt32 N; // number of perceptrons
    UInt32 theta; // training threshold
    std::vector<std::vector<SInt32>> perceptron_table;
    UInt64 history_reg;
    SInt32 last_y_predict;
};

#endif
