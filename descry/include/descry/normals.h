
#ifndef DESCRY_NORMALS_H
#define DESCRY_NORMALS_H

#include <descry/config/normals.h>
#include <descry/image.h>
#include <descry/viewer.h>

namespace descry {

class NormalEstimation {
public:
    bool configure(const Config& cfg);
    bool is_configured() const noexcept { return !!nest_; };
    DualNormals compute(const Image& image) const;
private:
    std::function<DualNormals(const Image&)> nest_;
    Viewer<Normals> viewer_;
    bool log_latency_ = false;
};

}

#endif //DESCRY_NORMALS_H
