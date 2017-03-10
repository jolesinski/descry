#ifndef DESCRY_CONFIG_CLUSTERS_H
#define DESCRY_CONFIG_CLUSTERS_H

namespace descry { namespace config { namespace clusters {

// Correspondence grouping

// PCL Hough Grouping
static constexpr auto HOUGH_TYPE = "uniform";
// Params
static constexpr auto BIN_SIZE = "bin-size";
static constexpr auto HOUGH_THRESH = "hough-threshold";
static constexpr auto USE_INTERP = "use-interpolation";
static constexpr auto USE_DIST_W = "use-distance-weight";

// PCL Geometric Consistency Grouping
static constexpr auto GEO_TYPE = "uniform";
// Params
static constexpr auto GC_SIZE = "gc-size";
static constexpr auto GC_THRESH = "gc-threshold";

}}}

#endif //DESCRY_CONFIG_CLUSTERS_H
