
#ifndef DESCRY_RECOGNIZER_H
#define DESCRY_RECOGNIZER_H

#include <descry/common.h>
#include <descry/model.h>

namespace descry {

class Recognizer {
public:
    bool configure(const Config& config);
    void train(const Model& model);
    std::vector<ModelInstance> recognize(const FullCloud::ConstPtr& scene);
};

}

#endif //DESCRY_RECOGNIZER_H
