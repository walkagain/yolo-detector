#include "config_parse.h"
#include <sstream>

void sspilt(std::string str, std::vector<std::string>&result, char separator=' ') {
    int pos = str.find(separator);
    result.clear();
    while(pos != std::string::npos) {
        result.emplace_back(str.substr(0, pos));
        str = str.substr(pos+1);
        pos = str.find(separator);
    }
    result.emplace_back(str);
    return ;
}

GlibParserHelper::GlibParserHelper(int argc, char** argv) {
    context = g_option_context_new("- yolo model detection application" );
    g_option_context_add_main_entries(context, entries, NULL);
    g_option_context_set_summary(context, "yolo application configure" );

    if  (!g_option_context_parse(context, &argc, &argv, &error)) {
        ERR_MSG_V("%s", error->message);
        g_option_context_free(context);
        exit (1);
    }

    if (cfg_files) {
      num_instances = g_strv_length (cfg_files);
    }

}
GlibParserHelper::~GlibParserHelper() {
    g_option_context_free(context);
}

bool GlibParserHelper::GetYoloConfig(Config& cfg, guint idx) {
    if (idx > num_instances) {
        g_print("file index [%u] out of num_instances[%u]\n", idx, num_instances);
        return false;
    }
    GKeyFile *cfg_file = g_key_file_new ();
    gboolean ret = FALSE;
    gchar **groups = NULL;
    gchar **group;
    if (!g_key_file_load_from_file (cfg_file, cfg_files[idx], G_KEY_FILE_NONE, &error)) {
        ERR_MSG_V("%s", error->message);
        g_key_file_free (cfg_file);
        return false;
    }
    groups = g_key_file_get_groups (cfg_file, NULL);
    for (group = groups; *group; group++) {
        if (!strncmp (*group, CONFIG_GROUP_PROPERTY, sizeof(CONFIG_GROUP_PROPERTY) - 1)) {
            if (!PropertyParse(cfg, cfg_file, *group)){
                g_key_file_free (cfg_file);
                return false;
            }
            break;
        }
    }
    g_key_file_free (cfg_file);
    return true;
}

bool GlibParserHelper::GetMeasureParams(MeasParams& params, guint idx) {
    if (idx > num_instances) {
        g_print("file index [%u] out of num_instances[%u]\n", idx, num_instances);
        return false;
    }
    GKeyFile *cfg_file = g_key_file_new ();
    gboolean ret = FALSE;
    gchar **groups = NULL;
    gchar **group;
    if (!g_key_file_load_from_file (cfg_file, cfg_files[idx], G_KEY_FILE_NONE, &error)) {
        ERR_MSG_V("%s", error->message);
        g_key_file_free (cfg_file);
        return false;
    }
    groups = g_key_file_get_groups (cfg_file, NULL);
    for (group = groups; *group; group++) {
        if (!strncmp (*group, CONFIG_GROUP_MEASUREMENT, sizeof(CONFIG_GROUP_MEASUREMENT) - 1)) {
            if (!MeasurementParse(params, cfg_file, *group)){
                g_key_file_free (cfg_file);
                return false;
            }
            break;
        }
    }
    g_key_file_free (cfg_file);
    return true;
}

bool GlibParserHelper::PropertyParse(Config& cfg, GKeyFile* key_file, gchar* group) {
    std::string g_str = std::string(group);
    if (g_str != CONFIG_GROUP_PROPERTY) {
        g_print("input configure group: %s, expected group:%s\n", group, CONFIG_GROUP_PROPERTY);
        return false;
    }
    gchar** keys = NULL;
	gchar** key = NULL;
    keys = g_key_file_get_keys(key_file, group, NULL, &error);
    if (!keys) {
        ERR_MSG_V("%s", error->message);
        return false;
    }
    std::stringstream ss;
    for (key = keys; *key; key++) {
        const gchar* value = g_key_file_get_string(key_file, group, *key, &error);
        if (std::string(*key) == PROPERTY_MODEL_FILE) {
            cfg.file_model_weights = std::string(value);
            g_print("model file: %s\n", cfg.file_model_weights.c_str());
        } else if (std::string(*key) == PROPERTY_NETWORK_FILE){
            cfg.file_model_cfg = std::string(value);
            g_print("configure file: %s\n", cfg.file_model_cfg.c_str());
        } else if (std::string(*key) == PROPERTY_NET_TYPE){
            cfg.net_type = (ModelType)(atoi(value));
            g_print("network type: %s\n", value);
        }else if (std::string(*key) == PROPERTY_NETWORK_MODE){
            cfg.inference_precison = (Precision)(atoi(value));
            g_print("inference_precison: %s\n", value);
        }else if (std::string(*key) == PROPERTY_CALIB_FILE){
            cfg.calibration_image_list_file_txt = std::string(value);
            g_print("calibration file: %s\n", cfg.calibration_image_list_file_txt.c_str());
        }else if (std::string(*key) == PROPERTY_DETECT_THRESH){
            cfg.detect_thresh = atof(value);
            g_print("detect thresh: %s\n", value);
        } else {
            g_print("UNKOWN KEY: %s\n", *key);
        }
    }
    return true;
}

bool GlibParserHelper::MeasurementParse(MeasParams& params, GKeyFile* key_file, gchar* group) {
    std::string g_str = std::string(group);
    if (g_str != CONFIG_GROUP_MEASUREMENT) {
        g_print("input configure group: %s, expected group:%s\n", group, CONFIG_GROUP_MEASUREMENT);
        return false;
    }
    gchar** keys = NULL;
	gchar** key = NULL;
    keys = g_key_file_get_keys(key_file, group, NULL, &error);
    if (!keys) {
        ERR_MSG_V("%s", error->message);
        return false;
    }
    std::stringstream ss;
    for (key = keys; *key; key++) {
        const gchar* value = g_key_file_get_string(key_file, group, *key, &error);
        if (std::string(*key) == MEASUREMENT_ENABLE) {
            params.enable = (bool)atoi(value);
            g_print("measurement enable: %d\n", params.enable);
        } else if (std::string(*key) == MEASUREMENT_STRATEGY) {
            params.strategy = atoi(value);
            g_print("strategy: %d\n", params.strategy);
        } else if (std::string(*key) == MEASUREMENT_MAX_STD){
            params.max_std = atof(value);
            g_print("max standard deviation: %f\n", params.max_std);
        } else if (std::string(*key) == MEASUREMENT_MAX_MEAN_DIFF){
            params.max_mean_diff = atof(value);
            g_print("max mean difference: %f\n", params.max_mean_diff);
        }else if (std::string(*key) == MEASUREMENT_AREA_PROPORTION){
            params.area_proportion = atof(value);
            g_print("area proportion: %f\n", params.area_proportion);
        }else if (std::string(*key) == MEASUREMENT_MAX_DEPTH){
            params.max_depth = atof(value);
            g_print("max depth: %f\n", params.max_depth);
        } else if (std::string(*key) == MEASUREMENT_ROI_DRAW) {
            params.roi_draw = bool(atoi(value));
            g_print("region of interest enable: %d\n", params.roi_draw);
        } else if (std::string(*key) == MEASUREMENT_GRID_WIDTH) {
            params.grid_w = atoi(value);
            std::cout << "grid width: " << params.grid_w << std::endl;
        } else if (std::string(*key) == MEASUREMENT_GRID_HEIGHT) {
            params.grid_h = atoi(value);
            std::cout << "grid height: " << params.grid_h << std::endl;

        } else if (std::string(*key) == MEASUREMENT_MIN_GRID) {
            params.min_grid = atoi(value);
            std::cout << "minimum grid size to remain: " << params.min_grid << std::endl;

        } else if (std::string(*key) == MEASUREMENT_COORD) {
            params.coord = (bool) atoi(value);
            std::cout << "coordinate show: " << params.coord << std::endl;

        } else if (std::string(*key) == MEASUREMENT_CONTOUR) {
            params.contour = (bool) atoi(value);
            std::cout << "contour show: " << params.coord << std::endl;

        } else if (std::string(*key) == MEASUREMENT_SIBLING_DIST) {
            params.sibling_dist = atof(value);
            std::cout << "sibling region distance: " << params.sibling_dist << std::endl;

        } else if (std::string(*key) == MEASUREMENT_TOP_REGION) {
            params.top_regions = atoi(value);
            std::cout << "top resions to choose: " << params.top_regions << std::endl;

        } else if (std::string(*key) == MEASUREMENT_SIBLING_MERGE) {
            params.sibling_merge = (bool)atoi(value);
            std::cout << "sibling region merge flag: " << params.sibling_merge << std::endl;

        } else if (std::string(*key) == MEASUREMENT_MAX_RECT) {
            params.max_rect = (bool)atoi(value);
            std::cout << "use maximum external rectangle: " << params.max_rect << std::endl;

        } else if (std::string(*key) == MEASUREMENT_USE_MIN_DEPTH) {
            params.use_min_depth = (bool)atoi(value);
            std::cout << "use minimum depth : " << params.use_min_depth << std::endl;
        }else {
            g_print("UNKOWN KEY: %s\n", *key);
        }
    }
    return true;
}

bool GlibParserHelper::ParseConfigFile (gchar *cfg_file_path) {
 // TODO: implement 
  return true;
}
