#include <catch.hpp>

#include <descry/recognizer.h>

TEST_CASE( "Configuration throws with malformed config", "[recognizer]" ) {
    descry::Recognizer rec;
    descry::Config cfg = YAML::Load("");

    REQUIRE_THROWS(rec.configure(cfg));
}