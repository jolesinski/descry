#ifndef DESCRY_PREPROCESS_H
#define DESCRY_PREPROCESS_H

#include <descry/common.h>
#include <descry/image.h>

namespace descry {

class Preprocess {
public:
    bool configure(const Config& cfg);

    void process(Image& image) const;
private:
    std::vector<std::function<void( Image& )>> steps_;
};

}

#endif //DESCRY_PREPROCESS_H
