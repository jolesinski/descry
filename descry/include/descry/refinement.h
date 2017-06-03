#ifndef DESCRY_REFINEMENT_H
#define DESCRY_REFINEMENT_H

#include <descry/common.h>
#include <descry/model.h>

#include <descry/config/refiner.h>

namespace descry {

class Refiner {
public:
    void configure(const Config& config);
    bool is_configured() const noexcept { return !!model_feed_; }
    void train(const Model& model);
    bool is_trained() const noexcept { return !!refiner_; }
    Instances compute(const Image& scene, const Instances& instances);
private:
    std::function<Instances( const Image&, const Instances& )> refiner_;
    std::function<decltype(refiner_)(const Model& model)> model_feed_;
};

}

#endif //DESCRY_REFINEMENT_H
