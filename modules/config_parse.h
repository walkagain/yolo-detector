/*************************************************************************
File Name: config_parse.c
Author: drew
Mail: xiaoshu.xie@ubtrobot.com 
Created Time: Thu 13 Aug 2020 11:08:01 AM CST
************************************************************************/
#ifndef __CONFIG_PARSE_H_
#define __CONFIG_PARSE_H_
#include <glib.h>
#include <glib/gprintf.h>
#include <locale.h>
#include "class_detector.h"
#include "depth_meas.h"

#define CONFIG_GROUP_PROPERTY "property"
#define CONFIG_GROUP_MEASUREMENT "measurement"
#define CONFIG_GROUP_SOURCE     "source"

#define PROPERTY_MODEL_FILE  "model-file"
#define PROPERTY_NETWORK_FILE  "network-config"
#define PROPERTY_CALIB_FILE  "calib-file"
#define PROPERTY_NETWORK_MODE  "network-mode"
#define PROPERTY_NET_TYPE  "net-type"
#define PROPERTY_DETECT_THRESH "detect-thresh"

#define MEASUREMENT_ENABLE      "enable"
#define MEASUREMENT_STRATEGY    "strategy"
#define MEASUREMENT_MAX_STD    "max_std"
#define MEASUREMENT_MAX_MEAN_DIFF    "max_mean_diff"
#define MEASUREMENT_AREA_PROPORTION   "area_proportion"
#define MEASUREMENT_MAX_DEPTH   "max_depth"
#define MEASUREMENT_ROI_DRAW    "roi_draw"

#define MEASUREMENT_GRID_HEIGHT     "grid_h"
#define MEASUREMENT_GRID_WIDTH      "grid_w"
#define MEASUREMENT_MIN_GRID        "min_grid"
#define MEASUREMENT_COORD           "coord"
#define MEASUREMENT_CONTOUR         "contour"
#define MEASUREMENT_SIBLING_DIST    "sibling_region_distance"
#define MEASUREMENT_TOP_REGION      "top_regions"
#define MEASUREMENT_SIBLING_MERGE   "sibling_merge"
#define MEASUREMENT_MAX_RECT        "max_rect"
#define MEASUREMENT_USE_MIN_DEPTH   "use_min_depth"


#define ERR_MSG_V(msg, ...) \
    g_print("** ERROR: <%s:%d>: " msg "\n", __func__, __LINE__, ##__VA_ARGS__)

static  gboolean beep = FALSE;
static gchar **cfg_files = NULL;

static  GOptionEntry entries[] =
{
    {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files, "specify path of config file", NULL},
    {NULL}
};

class GlibParserHelper{
public:
    GlibParserHelper(int argc, char** argv);
    ~GlibParserHelper();
    bool GetYoloConfig(Config& cfg, guint idx=0);
    bool GetMeasureParams(MeasParams& params, guint idx=0);
private:
    bool PropertyParse(Config& cfg, GKeyFile* key_file, gchar* group);
    bool MeasurementParse(MeasParams& params, GKeyFile* key_file, gchar* group);
    bool ParseConfigFile (gchar *cfg_file_path);
private:
    GOptionContext *context;
    guint num_instances;
    GError *error;
};

#endif  //__CONFIG_PARSE_H_
