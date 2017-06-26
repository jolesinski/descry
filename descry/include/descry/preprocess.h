#ifndef DESCRY_PREPROCESS_H
#define DESCRY_PREPROCESS_H

#include <descry/common.h>
#include <descry/image.h>
#include <descry/config/preprocess.h>

namespace descry {

class Preprocess {
public:
    bool configure(const Config& cfg);
    Image filter(const descry::FullCloud::ConstPtr& scene) const;
    void process(Image& image) const;
private:
    std::vector<std::function<Image( const Image& )>> filtering_;
    std::vector<std::function<void( Image& )>> steps_;
};

}

#endif //DESCRY_PREPROCESS_H
