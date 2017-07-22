#ifndef DESCRY_CONFIG_ALIGNER_H
#define DESCRY_CONFIG_ALIGNER_H

namespace descry { namespace config { namespace aligner {

static constexpr auto NODE_NAME = "aligner";

static constexpr auto DESCRIPTION_NODE = "description";

// Local features
static constexpr auto SPARSE_TYPE = "sparse";

// Template matching
static constexpr auto GLOBAL_TYPE = "global";
static constexpr auto SLIDING_TYPE = "sliding-window";
static constexpr auto MOCK_TYPE = "mock";

}}}

#endif //DESCRY_CONFIG_ALIGNER_H
