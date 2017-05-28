#ifndef DESCRY_VERIFICATION_H
#define DESCRY_VERIFICATION_H

#include <descry/common.h>
#include <descry/model.h>

#include <descry/config/verifier.h>

namespace descry {

class Verifier {
public:
    void configure(const Config& config);
    Instances compute(const Image& scene, const Instances& instances);
private:
    std::function<Instances( const Image&, const Instances& )> verifier_;
};

}

#endif //DESCRY_VERIFICATION_H
