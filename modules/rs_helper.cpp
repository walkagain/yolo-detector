/*************************************************************************
File Name: rs_helper.cpp
Author: drew
Mail: xiaoshu.xie@ubtrobot.com 
Created Time: Wed 12 Aug 2020 10:59:30 AM CST
************************************************************************/
#include "rs_helper.hpp"
#include <cstdlib>
#include <librealsense2/hpp/rs_frame.hpp>
using namespace rs2;

RealsenseHelper::RealsenseHelper() {
	pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
	profile = pipe.start(pipe_config);
    depth_clipping_distance = 1.f;

    depth_scale = GetDepthScale(profile.get_device());
    align = new rs2::align(RS2_STREAM_COLOR);
}

RealsenseHelper::~RealsenseHelper() {
    if (align) delete align;
}


float RealsenseHelper:: GetDepthScale() {
    return GetDepthScale(profile.get_device());
}

bool RealsenseHelper::SetDepthClippingDistance(float distance) {
    depth_clipping_distance = distance;
    return true;
}

bool RealsenseHelper::SetAlign(rs2_stream align_to) {
    if (align) delete align;
    align = new rs2::align(align_to);
    return true;
}
int RealsenseHelper::RealsenseAlign() {
	const char* depth_win="depth_Image";
    namedWindow(depth_win,WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);
	while (getWindowProperty(depth_win, WND_PROP_AUTOSIZE)&&getWindowProperty(color_win, WND_PROP_AUTOSIZE))
    {
        rs2::frameset frameset = pipe.wait_for_frames();
 
        // rs2::pipeline::wait_for_frames() can replace the device it uses in case of device error or disconnection.
        // Since rs2::align is aligning depth to some other stream, we need to make sure that the stream was not changed
        //  after the call to wait_for_frames();
        if (ProfileChanged(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            if (align) delete align;
            align = new rs2::align(RS2_STREAM_COLOR);
            depth_scale = GetDepthScale(profile.get_device());
        }
 
        //Get processed aligned frame
        auto processed = align->process(frameset);
 
        // Trying to get both other and aligned depth frames
        rs2::frame aligned_color_frame = processed.get_color_frame();//processed.first(align_to);
        rs2::frame aligned_depth_frame = processed.get_depth_frame().apply_filter(c);
 
        rs2::frame before_depth_frame=frameset.get_depth_frame().apply_filter(c);
        const int depth_w=aligned_depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h=aligned_depth_frame.as<rs2::video_frame>().get_height();
        const int color_w=aligned_color_frame.as<rs2::video_frame>().get_width();
        const int color_h=aligned_color_frame.as<rs2::video_frame>().get_height();
        const int b_color_w=before_depth_frame.as<rs2::video_frame>().get_width();
        const int b_color_h=before_depth_frame.as<rs2::video_frame>().get_height();
        //If one of them is unavailable, continue iteration
        if (!aligned_depth_frame || !aligned_color_frame)
        {
            continue;
        }
        Mat aligned_depth_image(Size(depth_w,depth_h),CV_8UC3,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        Mat aligned_color_image(Size(color_w,color_h),CV_8UC3,(void*)aligned_color_frame.get_data(),Mat::AUTO_STEP);
        Mat before_color_image(Size(b_color_w,b_color_h),CV_8UC3,(void*)before_depth_frame.get_data(),Mat::AUTO_STEP);


		//display
        imshow(depth_win,aligned_depth_image);
        imshow(color_win,aligned_color_image);
		imshow("depth_before_align",before_color_image);
        if ((char)waitKey(10) == 'q') break;
    }
	return EXIT_SUCCESS;
}
bool RealsenseHelper::AlignedFrameData(FrameData& data) {
     rs2::frameset frameset = pipe.wait_for_frames();
     if (ProfileChanged(pipe.get_active_profile().get_streams(), profile.get_streams()))
     {
        //If the profile was changed, update the align object, and also get the new device's depth scale
        profile = pipe.get_active_profile();
        if (align) delete align;
        align = new rs2::align(RS2_STREAM_COLOR);
        depth_scale = GetDepthScale(profile.get_device());
    }

    //Get processed aligned frame
    auto processed = align->process(frameset);

    // Trying to get both other and aligned depth frames
    rs2::frame aligned_color_frame = processed.get_color_frame();//processed.first(align_to);
    rs2::frame aligned_depth_frame = processed.get_depth_frame().apply_filter(c);
    if (!aligned_depth_frame || !aligned_color_frame) return false;

    const int depth_w=aligned_depth_frame.as<rs2::video_frame>().get_width();
    const int depth_h=aligned_depth_frame.as<rs2::video_frame>().get_height();
    const int color_w=aligned_color_frame.as<rs2::video_frame>().get_width();
    const int color_h=aligned_color_frame.as<rs2::video_frame>().get_height();
    data.rgb =  cv::Mat(Size(color_w,color_h),CV_8UC3,(void*)aligned_color_frame.get_data(),cv::Mat::AUTO_STEP);
    data.depth = cv::Mat(Size(depth_w,depth_h),CV_8UC3,(void*)aligned_depth_frame.get_data(),cv::Mat::AUTO_STEP);
    auto d_tmp = processed.get_depth_frame();
    GetTotalDepth(d_tmp, data.depth_map);
    data.color_intr = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    return true;
}

float RealsenseHelper::GetSingleDistance(rs2::frame& depth, int row, int col) {
    rs2::depth_frame rs_depth(depth);
    return rs_depth.get_distance(row, col); //check out of range?
}

vector<float> RealsenseHelper::GetMultiDistances(rs2::frame& depth, vector<tuple<int, int> >& points) {
    rs2::depth_frame rs_depth(depth);
    vector<float> dists;
    for (auto& p: points) {
        dists.push_back(rs_depth.get_distance(std::get<0>(p), std::get<1>(p)));
    }
    return dists;
}

void RealsenseHelper::GetTotalDepth(rs2::frame& depth, vector<vector<float> >& real_depth) {
    const int depth_w=depth.as<rs2::video_frame>().get_width();
    const int depth_h=depth.as<rs2::video_frame>().get_height();
    real_depth.clear();
    real_depth.resize(depth_h);
    rs2::depth_frame rs_depth(depth);
    for (int i=0; i < depth_h; i++) {
        for (int j = 0; j < depth_w; j++) {
            real_depth[i].push_back(rs_depth.get_distance(j,i));
        }
    }
    return;
}

void RealsenseHelper::GetLocalDepth(rs2::frame& depth, vector<vector<float> >& bbox_depth, int x, int y, int width, int height) {
    if (x < 0 || y < 0) return ;
    const int depth_w=depth.as<rs2::video_frame>().get_width();
    const int depth_h=depth.as<rs2::video_frame>().get_height();
    if (width > depth_w || height > depth_h) return ;
    bbox_depth.clear();
    bbox_depth.resize(height);
    rs2::depth_frame rs_depth(depth);
    for (int i = y; i < y+height; i++) {
        for (int j = x; j < x+width; j++) {
            bbox_depth[i].emplace_back(rs_depth.get_distance(j,i));
        }
    }
    return ;
}

bool RealsenseHelper::ProfileChanged(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev) {
	for (auto&& sp : prev) {
		//If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;	
}

float RealsenseHelper::GetDepthScale(rs2::device dev) {
	// Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}
