#include <catch.hpp>

#include <descry/recognizer.h>

TEST_CASE( "Configuring recognizer fails with empty config", "[recognizer]" ) {
    descry::Recognizer rec;
    descry::Config cfg = YAML::Load("");

    REQUIRE(!rec.configure(cfg));
}