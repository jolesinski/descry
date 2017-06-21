#include <descry/common.h>
#include <descry/latency.h>
#include <descry/test/config.h>
#include <descry/willow.h>

void recognize(const descry::Config& cfg) {
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = cfg[descry::config::SCENE_NODE].as<std::string>();
    auto model_name = cfg[descry::config::MODEL_NODE].as<std::string>();
    auto latency = descry::measure_latency("Loading test set");
    auto test_data = willow.loadSingleTest(test_name);
    latency.restart("Loading model");
    auto model = willow.loadModel(model_name);
    latency.finish();
}

int main(int argc, char * argv[]) {
    descry::logger::init();
    auto g_log = descry::logger::get();
    if (!g_log)
        return EXIT_FAILURE;

    auto cfg = descry::Config{};
    try {
        if (argc > 1)
            cfg = YAML::LoadFile(argv[1]);
    } catch (YAML::ParserException& error) {
        g_log->error("Parser error: {}", error.what());
    } catch (YAML::BadFile& error) {
        g_log->error("Bad config file: {}", error.what());
    }

    if (cfg.IsNull()) {
        g_log->error("Invalid config");
        return EXIT_FAILURE;
    }

    recognize(cfg);

    return EXIT_SUCCESS;
}
