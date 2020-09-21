/*************************************************************************
File Name: rs_helper.hpp
Author: drew
Mail: xiaoshu.xie@ubtrobot.com 
Created Time: Wed 12 Aug 2020 10:42:45 AM CST
************************************************************************/
#ifndef  __RS_HELPER_H_
#define  __RS_HELHER_H_
#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
 
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>


using namespace std;
using namespace cv;

struct FrameData {
	cv::Mat rgb;
	cv::Mat depth;
	//rs2::frame rs_depth;
	vector<vector<float> >depth_map;
	rs2_intrinsics color_intr;
};

class RealsenseHelper {
public:
	RealsenseHelper();
	~RealsenseHelper();
	bool SetDepthClippingDistance(float distance);
	bool AlignedFrameData(FrameData& data);
	float GetDepthScale();
	int RealsenseAlign();
	bool SetAlign(rs2_stream align_to=RS2_STREAM_COLOR);
	float GetSingleDistance(rs2::frame& depth, int row, int col);
	vector<float>GetMultiDistances(rs2::frame& depth, vector<tuple<int, int> >& points);
	void GetTotalDepth(rs2::frame& depth, vector<vector<float> >& real_depth);
	void GetLocalDepth(rs2::frame& depth, vector<vector<float> >& bbox_depth, int x, int y, int width, int height);
private:
	bool ProfileChanged(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);
	float GetDepthScale(rs2::device dev);


private:
	rs2::pipeline pipe;
    rs2::config pipe_config;
	rs2::pipeline_profile profile;
	float depth_clipping_distance;
	rs2::colorizer c;   // Helper to colorize depth images
	rs2::align* align;
	float depth_scale;
};



#endif  //__RS_HELPER_H_


