#ifndef DESCRY_DESCRIBER_H
#define DESCRY_DESCRIBER_H

#include <descry/image.h>
#include <descry/config/descriptors.h>

namespace descry {

template <class Descriptor>
class Describer {
public:
    bool configure(const Config &config);
    cupcl::DualContainer<Descriptor> compute(const Image& image);

private:
    std::function<cupcl::DualContainer<Descriptor>( const Image& )> _descr;
};

}

#endif //DESCRY_DESCRIBER_H