
#ifndef DESCRY_MODEL_H
#define DESCRY_MODEL_H

#include <descry/projector.h>
#include <descry/preprocess.h>

namespace descry {

class Model
{
public:
    template<class Descriptor>
    using Description = cupcl::DualContainer<Descriptor>;

    Model(const FullCloud::ConstPtr& full, const Projector& projector);
    Model(const FullCloud::ConstPtr& full, AlignedVector<View>&& views);

    const FullCloud::ConstPtr& getFullCloud() const { return full_; }
    const AlignedVector<View>& getViews() const { return views_; }
    AlignedVector<View>& getViews() { return views_; }

    void prepare(const Preprocess& preprocessor);
protected:
    FullCloud::ConstPtr full_;
    AlignedVector<View> views_;
};

}

#endif //DESCRY_MODEL_H
