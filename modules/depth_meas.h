#ifndef __DEPTH_MEASUREMENT_H_
#define __DEPTH_MEASUREMENT_H_
#include <iostream>
#include <vector>

struct MeasParams{
    bool enable = true;
    int strategy=0;
    float max_std;
    float max_mean_diff;
    float area_proportion;
    float max_depth;
    bool roi_draw = false;
    int grid_h = 10;
    int grid_w = 10;
    int min_grid = 5;
    bool coord = false;
    bool contour=false;
    int top_regions = 5;
    float sibling_dist = 0.1;
    bool sibling_merge = false;
    bool max_rect = false;
    bool use_min_depth = false;
};

#endif  //__DEPTH_MEASUREMENT_H_