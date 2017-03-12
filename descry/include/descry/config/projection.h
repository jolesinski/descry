#ifndef DESCRY_CONFIG_PROJECTION_H
#define DESCRY_CONFIG_PROJECTION_H

namespace descry { namespace config { namespace projection {

// Willow-like database loader
static constexpr auto WILLOW_TYPE = "willow";
// Params (required):
static constexpr auto VIEWS_PATH = "views-path";

// Spiral tesselation projector
static constexpr auto SPIRAL_TYPE = "spiral";
// Params (required):
static constexpr auto SPHERE_RAD = "sphere-radius";
static constexpr auto SPIRAL_TURNS = "spiral-turns";
static constexpr auto SPIRAL_DIV = "spiral-divisions";

}}}

#endif //DESCRY_CONFIG_PROJECTION_H
