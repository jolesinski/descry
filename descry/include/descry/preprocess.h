#ifndef DESCRY_PREPROCESS_H
#define DESCRY_PREPROCESS_H

#include <descry/common.h>
#include <descry/image.h>
#include <descry/config/preprocess.h>

namespace descry {

class Preprocess {
public:
    void configure(const Config& cfg);
    void process(Image& image) const;
private:
    std::vector<std::function<void( Image& )>> steps_;
    bool log_latency_;
};

}

#endif //DESCRY_PREPROCESS_H
