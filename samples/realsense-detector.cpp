#include "class_timer.hpp"
#include "class_detector.h"
#include "rs_helper.hpp"
#include "config_parse.h"
#include "profile_detect.h"

#include <memory>
#include <thread>
#include <math.h>
#include <limits>
#include <stdio.h>

const std::vector<string>labels{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", 
"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
"donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", 
"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
"teddy bear", "hair drier", "toothbrush"};

const std::map<int, string>detClasses{{0, "person"}};

struct areaInfo {
	int x;
	int y;
	float depth;	//mean depth
	float min_depth;
	float max_depth;
	cv::Rect roi;
	vector<tuple<int, int> >contours;
	areaInfo& operator= (const areaInfo& other) {
		x = other.x;
		y = other.y;
		depth = other.depth;
		roi = other.roi;
		min_depth = other.min_depth;
		////////////////////////////////
		contours.clear();
		for (int i=0; i < other.contours.size(); i++) contours.emplace_back(other.contours[i]);
		return *this;
	}

	bool operator>(const areaInfo& other) {
		return contours.size() > other.contours.size();
	}

	bool operator<(const areaInfo& other) {
		return contours.size() < other.contours.size();
	}
};
struct estimationItem {
	float mean = 0.0;
	float std = 0.0;
	float ame = 0.0;	//absolute mean error
	int center_x = 0;
	int center_y = 0;
	float min_d = std::numeric_limits<float>::max();
	float max_d = std::numeric_limits<float>::min();
};

int g_width = 0;
int g_height = 0;

/************************************************************************
 * Add depth information to detect objct
 * 
 * @param data		: Input frame information, inlcude rgb and depth
 * @param detRes	: yolovx detect result
 * @param params	: depth measure parameters 
*************************************************************************/
bool FrameRender(FrameData& data, BatchResult& detRes, const MeasParams& params);

/************************************************************************
 * get distance of bound box certer point
 * 
 * @param depth		: depth data of input image
 * @param x			: top left x coordinate of bbox
 * @param y			: top left y coordinate of bbox
 * @param w			: width of bbox
 * @param h			: height of bbox
*************************************************************************/
areaInfo getCenterDistance(vector<vector<float> >& depth, int x, int y, int w, int h);

/************************************************************************
 * split bbox into multi continuous regions, and get distance of the largest one
 * 
 * @param depth		: depth data of input image
 * @param x			: top left x coordinate of bbox
 * @param y			: top left y coordinate of bbox
 * @param w			: width of bbox
 * @param h			: height of bbox
 * @param params	: depth measure parameters
 * @return areaInfo : point information to represent the region
*************************************************************************/
areaInfo getLargestAreaDistance(vector<vector<float> >& depth, int x, int y, int w, int h, const MeasParams& params);

/************************************************************************
 * calculate estimation items(mean, standard devaition, et al.) for depth block
 * 
 * @param data		: depth data of input image
 * @param item		: store calculate result
*************************************************************************/
void calcEstimationItems(vector<vector<float> >& data, estimationItem& item);

/************************************************************************
 * make neighbour blocks into regions 
 * 
 * @param blocks_in		: blocks of detect bound box
 * @param out			: splited result with set of block index
 * @param params		: depth measure parameters
*************************************************************************/
void getContinuousBlocks(vector<vector<estimationItem> >& blocks_in, vector<vector<tuple<int, int> > >& out, const MeasParams& params);

/************************************************************************
 * start with a block, travel input blocks to search continuous blocks by depth first search
 * 
 * @param blocks_in		: blocks of detect bound box
 * @param visited		: label if some block is put into a set
 * @param x				: row coordinate of current block
 * @param y				: column coordinate of current block 
 * @param adjacencies	: current search result
 * @param params		: depth measure parameters
*************************************************************************/
void blocksSearch(vector<vector<estimationItem> >& blocks_in, vector<vector<bool> >& visited, int x, int y, \
vector<tuple<int, int>>& adjacencies, const MeasParams& params);

/************************************************************************
 * check area proportion for bbox vs input image
 * 
 * @param bw			: width of bbox
 * @param bh			: height of bbox
 * @param w				: width of input image
 * @param h				: height of input image 
 * @param params		: depth measure parameters
*************************************************************************/
bool areaProportionValid(int bw, int bh, int w, int h, const MeasParams& params);

/************************************************************************
 * get depth data for specified block
 * 
 * @param depth			: depth data of input image
 * @param x_start		: start position of image row
 * @param y_start		: start position of image column
 * @param x_step		: width of block
 * @param y_step		: height of block
 * @param block			: store block data
 * @param params		: depth measure parameters
*************************************************************************/
void getBlockData(vector<vector<float> >& depth, int x_start, int y_start, int x_step, int y_step, vector<vector<float> >& block, MeasParams& params);

/************************************************************************
 * draw contour of detect object
 * 
 * @param img			: input image
 * @param cells			: cells in object region
 * @param w				: width of cell
 * @param h				: height of cell
*************************************************************************/
void drawContours(cv::Mat& img, vector<tuple<int, int> >& cells, int w, int h);

/************************************************************************
 * top N regions by comparing region area
 * 
 * @param regions		: input regions, consist of blocks
 * @param indexs		: index list for topN regions 
 * @param n				: 
*************************************************************************/
void topRegions(vector<vector<tuple<int, int> > >& regions, vector<int>& indexs, int n = 5);
/************************************************************************
 * merge possible sibling regions into bigger region
 * 
 * @param items			: measurement info of all blocks
 * @param regions		: input regions, consist of blocks
 * @param inIdxs		: index list of specified regions
 * @param outReg		: informations list of merged regions
*************************************************************************/
void mergeSiblingRegions(vector<vector<estimationItem>>&items, vector<vector<tuple<int, int>>>& regions, vector<int>& inIdxs, vector<areaInfo>& outReg);

/********************************************
 * tools function
*********************************************/
void regionStatistic(vector<vector<estimationItem>>&items, vector<tuple<int, int>>& region, areaInfo& info);
bool isSiblingArea(const areaInfo& area_a, const areaInfo& area_b);

int main(int argc, char** argv) {
	GlibParserHelper parser(argc, argv);
	Config config;
	parser.GetYoloConfig(config, 0);
	MeasParams params;
	parser.GetMeasureParams(params, 0);
	if (params.strategy == 1) {
		printf("compute largest continuous area depth\n");
	} else {
		printf("compute bbox center point depth\n");
	}
	RealsenseHelper rs;
	std::chrono::steady_clock clock;
	uint64_t frame_id = 0;

	FrameData data;
	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config);
	std::vector<BatchResult> batch_res;
	auto start = clock.now();
	while (true) {
		if (! rs.AlignedFrameData(data)) continue;
		frame_id ++;
		if (g_width == 0) {
			g_width = data.rgb.cols;
			g_height = data.rgb.rows;
			std::cout << "width: " << g_width << ", height: " << g_height << std::endl;
		}

		std::vector<cv::Mat> batch_img;
		batch_img.emplace_back(data.rgb);
		detector->detect(batch_img, batch_res);
		
		if (batch_res.empty()) continue;
		FrameRender(data, batch_res[0], params);
		auto duration = static_cast<chrono::duration<double>>(clock.now() - start);
		float frameRate = frame_id/duration.count();
		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << frameRate << " fps";
		cv::putText(data.rgb, ss.str(), Point(0,30), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
		cv::imshow("color", data.rgb);
		cv::imshow("depth", data.depth);
		if ((char)cv::waitKey(1) == 'q') break;
	}
	return 0;
}

bool FrameRender(FrameData& data, BatchResult& detRes, const MeasParams& params) {

	ProfileDetHelper profile;
	for (const auto &r: detRes) {
		//if (r.id != 0) continue;
		if (detClasses.find(r.id) == detClasses.end()) continue;
		cv::rectangle(data.rgb, r.rect, cv::Scalar(255, 0, 0), 2);
		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << labels[r.id] << "  " << r.prob;
		cv::putText(data.rgb, ss.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
		
		ss.str("");
		if (params.enable) {
			areaInfo p;
			if (params.strategy == 1 && areaProportionValid(r.rect.width, r.rect.height, g_width, g_height, params)) { 
				p = getLargestAreaDistance(data.depth_map, r.rect.x, r.rect.y, r.rect.width, r.rect.height, params);
				if (params.roi_draw) {
					cv::rectangle(data.rgb, p.roi, cv::Scalar(0, 255, 0), 2);
				}
			} else {
				p = getCenterDistance(data.depth_map, r.rect.x, r.rect.y, r.rect.width, r.rect.height);
			}
			if (params.contour) drawContours(data.rgb, p.contours, params.grid_w, params.grid_h);
			if (params.coord) {
				float planarPoint3d[3];
				float uvpixel[2] = {(float)p.x, (float)p.y};
				rs2_deproject_pixel_to_point(planarPoint3d, &(data.color_intr), uvpixel, p.depth);
				ss << std::fixed << std::setprecision(3) << "(" << planarPoint3d[0] << "," << planarPoint3d[1] << "," << planarPoint3d[2] << ")";
			} else {
				ss << std::fixed << std::setprecision(3) << p.depth << "m";
			}
			cv::putText(data.rgb, ss.str(), cv::Point(p.x-10, p.y + 10), 0, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		#if 0
		//profile detect
		vector<vector<cv::Point> >contours;
		vector<cv::Vec4i> hierarchy;
		profile.ProfileDetect(data.rgb, r.rect.x, r.rect.y, r.rect.width, r.rect.height, contours, hierarchy);

		vector<int> indexs;
		profile.ContoursTopN(contours, indexs);
		//vector<cv::Vec4i> empt();
		profile.ContoursDraw(data.rgb, contours, hierarchy);
		//profile.ContoursDrawByIndexs(img, contours, indexs);
		#endif
	}
	return true;
}

areaInfo getCenterDistance(vector<vector<float> >& depth, int x, int y, int width, int height) {
	int center_x = x + width / 2;
	int center_y = y + height / 2;
	if (center_y < 0 || center_y >= depth.size() || center_x < 0 || center_x >= depth[0].size()) {
		std:cerr << "invalid box bound: " << x << ", " << y << ", " << width << ", " << height << std::endl;
		return {0, 0, 0.0}; 
	}
	float d =  depth[center_y][center_x];
	//if center point  is abnormal, get average value among 5 x 5 grid
	if (d <= 0.0 || isnan(d) || isinf(d)) {
		d = 0.0;
		int x_l = center_x - 5 >= x?center_x - 5:x;
		int x_r = center_x + 5 < x + width?center_x - 5:x + width;
		int y_l = center_y - 5 >= y? center_y - 5:y;
		int y_r = center_y - 5 < y+height? center_y + 5:y+height;
		int count = 0;
		for (int i = y_l; i < y_r; i++) {
			for (int j = x_l; j < x_r; j++) {
				if (depth[i][j] <= 0.0 || isnan(depth[i][j]) || isinf(depth[i][j])) continue;
				d += depth[i][j];
				count ++;
			}
		}
		if (count != 0) d /= count;
	}
	return {center_x, center_y, d};
}

areaInfo getLargestAreaDistance(vector<vector<float> >& depth, int x, int y, int w, int h, const MeasParams& params) {
	if (depth.empty()) return {0, 0, 0.0};
	int grid_x = w / params.grid_w;
	int grid_y = h / params.grid_h;

	int x_step = params.grid_w;
	int y_step = params.grid_h;

	bool w_discard = true, h_discard = true;
	int w_remain = w % x_step;
	int h_remain = h % y_step;

	if (w_remain > params.min_grid) {
		w_discard = false;
		grid_x ++;
	}
	if (h_remain >= params.min_grid) {
		h_discard = false;
		grid_y ++;
	}

	vector<vector<estimationItem> >items(grid_y, vector<estimationItem>(grid_x, {0.0}));
	vector<vector<float> >block;

	block.resize(y_step);
	for (int i=0; i < y_step; i++) {
		block[i].resize(x_step, 0.0);
	}

	int calc_w = x_step * grid_x, calc_h = y_step * grid_y;

	int x_start = x, y_start = y;
	//float MaxDepth = std::numeric_limits<float>::min();
	for (int i = 0; i < grid_y; i++) {
		for (int j=0; j < grid_x; j++) {
			int k = y_start, p;
			if (!w_discard && j == grid_x - 2) {
				x_step = w_remain;
			}
			// reset the block 
			block.clear();
			block.resize(y_step);
			for (int i=0; i < y_step; i++) {
				block[i].resize(x_step, 0.0);
			}
			//get block values by specified steps
			for (; k < y_start + y_step; k++) {
				p = x_start;
				for (; p < x_start+x_step; p++) {
					if (isinf(depth[k][p]) || depth[k][p] > params.max_depth) depth[k][p] = params.max_depth;
					if (isnan(depth[k][p]) || depth[k][p] < 0.0) depth[k][p] = 0.0;
					block[k-y_start][p-x_start] = depth[k][p];
				}
				x_start = p - x_step;
			}
			//assistance
			items[i][j].center_x = p - x_step / 2;
			items[i][j].center_y = k - y_step / 2;
			// set the x_start and y_start to beginning index of next grid block 
			// y represents row, x represents column
			if (j == grid_x -1) {
				x_start = x;
				y_start = k;
			} else {
				y_start = k - y_step;
				x_start = p;
			}
			calcEstimationItems(block, items[i][j]);
		}
		if (!h_discard && i == grid_y - 2) {
			y_step = h_remain;
		}
		x_step = params.grid_w;
	}
	vector<vector<tuple<int, int>>> out;
	getContinuousBlocks(items, out, params);

	int best_idx = 0, bcount = 0, total = 0;
	vector<int> regIdxs;
	topRegions(out, regIdxs, params.top_regions);
	total = grid_x * grid_y;

	areaInfo p;

	if (!params.sibling_merge) {
		if (params.max_rect) {
			for (int i=0; i < regIdxs.size(); i++) {
				areaInfo candidate;
				regionStatistic(items, out[regIdxs[i]], candidate);
				if ((p.roi.width * p.roi.height) < (candidate.roi.width * candidate.roi.height)) {
					p = candidate;
					bcount = out[regIdxs[i]].size();
				}
			}
		} else {
			best_idx = regIdxs[0];	//regIdxs should always not empty
			bcount = out[best_idx].size();
			regionStatistic(items, out[best_idx], p);
		}
		//printf("total blocks:%d, choose blocks:%d, proportion:%.2f\n", total, bcount, ((float)bcount) / total);
	} else {
		vector<areaInfo>regInfo;
		mergeSiblingRegions(items, out, regIdxs, regInfo);
		int choose_idx = -1, area = 0;
		for (int i=0; i < regInfo.size(); i++) {
			if (regInfo[i].roi.width * regInfo[i].roi.height > area) {
				area = regInfo[i].roi.width * regInfo[i].roi.height;
				choose_idx = i;
			}
		}
		if (choose_idx != -1) p = regInfo[choose_idx];
		else std::cerr << "no merge region match" << std::endl;
	}
	if (params.use_min_depth) p.depth = p.min_depth;
	return p;
}

void calcEstimationItems(vector<vector<float> >& data, estimationItem& item) {
	float sum_ = 0.0;
	int count = 0;
	item.mean = 0.0, item.std = 0.0;
	for (auto& d: data) {
		for (float d_: d) {
			if (d_ <= 0.0 || isnan(d_) || isinf(d_)) continue;
			count ++;
			sum_ += d_;
			if (d_ < item.min_d && d_ != 0.0 ) item.min_d = d_;
			if (d_ > item.max_d && d_ != 0.0 ) item.max_d = d_;
		}
	}
	if (count != 0) {
		item.mean = sum_ / count;
	} else {
		//printf("no data input\n");
		return ;
	}
	sum_ = 0.0;
	float absum = 0.0;
	for (auto& d: data) {
		for (float d_: d) {
			if (d_ < 0.0 || isnan(d_) || isinf(d_)) continue;
			sum_ += pow(d_ - item.mean, 2.0);
			absum += abs(d_ - item.mean);
		}
	}
	item.std = sqrt(sum_ / count);
	item.ame = absum / count;
	return ;
}

void getContinuousBlocks(vector<vector<estimationItem> >& blocks_in, vector<vector<tuple<int, int> > >& out, const MeasParams& params) {
	if (blocks_in.empty()) {
		std::cerr << "empty input blocks" << std::endl;
		return;
	}
	int d1_size = blocks_in.size();
	int d2_size =  blocks_in[0].size();

	vector<vector<bool> >visited(d1_size, vector<bool>(d2_size, false));
	for (int i=0; i < d1_size; i++) {
		for (int j=0; j < d2_size; j++) {
			if(visited[i][j]) continue;
			vector<tuple<int, int> > adjacencies;
			blocksSearch(blocks_in, visited, i, j, adjacencies, params);
			if (! adjacencies.empty()) out.emplace_back(adjacencies);
		}
	}

	return ;
}

void blocksSearch(vector<vector<estimationItem> >& blocks_in, vector<vector<bool> >& visited, int x, int y, \
vector<tuple<int, int>> & adjacencies, const MeasParams& params) {
	if (blocks_in.empty()) {
		std::cerr << "empty input blocks" << std::endl;
		return;
	}
	int d1_size = blocks_in.size();
	int d2_size =  blocks_in[0].size();
	if (x < 0 || x >= d1_size || y < 0 || y >= d2_size) return;
	if (visited[x][y]) return;
	visited[x][y] = true;
	adjacencies.emplace_back(std::make_tuple(x, y));
	if (blocks_in[x][y].std > params.max_std || blocks_in[x][y].mean == 0.0) return ;
	vector<tuple<int, int> >directs{{0,1}, {0,-1}, {1,0}, {-1, 0}, {1, -1}, {-1, 1}, {1, 1}, {-1, -1}};
	for (auto& d: directs) {
		int next_x = x + std::get<0>(d);
		int next_y = y + std::get<1>(d);
		if (next_x < 0 || next_x >= d1_size || next_y < 0 || next_y >= d2_size) continue;

		//judge if block is continuous one
		if (abs(blocks_in[x][y].mean - blocks_in[next_x][next_y].mean) > params.max_mean_diff \
		|| blocks_in[next_x][next_y].std > params.max_std) continue;
		blocksSearch(blocks_in, visited, next_x, next_y, adjacencies, params);
	}
	return ;
}

bool areaProportionValid(int bw, int bh, int w, int h, const MeasParams& params) {
	return (1.0 * bw * bh)/(w*h) > params.area_proportion;
}

void getBlockData(vector<vector<float> >& depth, int x_start, int y_start, int x_step, int y_step, vector<vector<float> >& block, MeasParams& params) {
	int k = y_start, p;
	for (; k < y_start + y_step; k++) {
		p = x_start;
		for (; p < x_start+x_step; p++) {
			if (isinf(depth[k][p]) || depth[k][p] > params.max_depth) depth[k][p] = params.max_depth;
			if (isnan(depth[k][p]) || depth[k][p] < 0.0) depth[k][p] = 0.0;
			block[k-y_start][p-x_start] = depth[k][p];
		}
		x_start = p - x_step;
	}
	return ;
}

void drawContours(cv::Mat& img, vector<tuple<int, int> >& cells, int w, int h) {
	for (int i=0; i < cells.size(); i++) {
		int tl_x = std::get<0>(cells[i]) - w / 2;
		int tl_y = std::get<1>(cells[i]) - h / 2;
		int br_x = std::get<0>(cells[i]) + w / 2;
		int br_y = std::get<1>(cells[i]) + h / 2;
		for (int row = tl_y; row < br_y; row ++) {
			for (int col = tl_x; col < br_x; col++) {
				*(img.data + img.step[0] * row + img.step[1] * col + img.elemSize1() * 0) = 128;
				*(img.data + img.step[0] * row + img.step[1] * col + img.elemSize1() * 1) = 255;
				*(img.data + img.step[0] * row + img.step[1] * col + img.elemSize1() * 2) = 128;
			}
		}
	}
	return ;
}

void topRegions(vector<vector<tuple<int, int> > >& regions, vector<int>& indexs, int n) {
	indexs.clear();
	vector<int> tmp(regions.size(), -1);
	n = n <= regions.size()? n:regions.size();
	int best_ind = -1, mcnt = 0;
	for (int i=0; i < n; i++) {
		for (int j=0; j < regions.size(); j++) {
			//note: mcnt must not be negative
			if ((regions[j].size() > mcnt) && (tmp[j] == -1)) {
				mcnt = regions[j].size();
				best_ind = j;
			}
		}
		//std::cout << mcnt << " ";
		tmp[best_ind] = best_ind;
		indexs.emplace_back(best_ind);
		best_ind = -1;
		mcnt = 0;
	}
	//std::cout << std::endl;
	return ;
}

void mergeSiblingRegions(vector<vector<estimationItem>>&items, vector<vector<tuple<int, int>>>& regions,\
vector<int>& inIdxs, vector<areaInfo>& outReg) {
	outReg.clear();
	vector<tuple<int, areaInfo>>regInfo;
	for (const auto& idx: inIdxs) {
		areaInfo p;
		regionStatistic(items, regions[idx], p);
		regInfo.emplace_back(std::make_tuple(idx, p));
	}
	vector<vector<int> >mergeIdx;
	for (int i=0; i < regInfo.size(); i++) {
		if (std::get<0>(regInfo[i]) == -1) continue;
		areaInfo mergeRegion = std::get<1>(regInfo[i]);
		int blk_num = regions[std::get<0>(regInfo[i])].size();
		for (int j=i+1; j < regInfo.size(); j++) {
			if (std::get<0>(regInfo[j]) == -1) continue;
			//TODO: optimize similar region judgement strategy
			if (isSiblingArea(mergeRegion, std::get<1>(regInfo[j]))) {
				areaInfo& j_point = std::get<1>(regInfo[j]);
				int j_blks = regions[std::get<0>(regInfo[j])].size();
				mergeRegion.depth = (mergeRegion.depth * blk_num + j_point.depth * j_blks) / (blk_num + j_blks);
				blk_num += j_blks;
				if (mergeRegion.min_depth > j_point.min_depth) mergeRegion.min_depth = j_point.min_depth;
				if (mergeRegion.max_depth > j_point.max_depth) mergeRegion.max_depth = j_point.max_depth;
				int tl_x = mergeRegion.roi.x > j_point.roi.x?j_point.roi.x:mergeRegion.roi.x;
				int tl_y = mergeRegion.roi.y > j_point.roi.y?j_point.roi.y:mergeRegion.roi.y;
				int br_x = mergeRegion.roi.x + mergeRegion.roi.width > j_point.roi.x + j_point.roi.width?\
				j_point.roi.x + j_point.roi.width:mergeRegion.roi.x + mergeRegion.roi.width;
				int br_y = mergeRegion.roi.y + mergeRegion.roi.height > j_point.roi.y + j_point.roi.height?\
				j_point.roi.y + j_point.roi.height:mergeRegion.roi.y + mergeRegion.roi.height;

				mergeRegion.roi.x = tl_x;
				mergeRegion.roi.y = tl_y;
				mergeRegion.roi.width = br_x - tl_x;
				mergeRegion.roi.height = br_y - tl_y;
				mergeRegion.x = (tl_x + br_x) / 2;
				mergeRegion.y = (tl_y + br_y) / 2;

				for (auto& cntr: j_point.contours) mergeRegion.contours.emplace_back(cntr);
				std::get<0>(regInfo[j]) = -1;
			}
		}	
		outReg.emplace_back(mergeRegion);
	}
	sort(outReg.begin(), outReg.end(), [](const areaInfo& a, const areaInfo& b){return a.contours.size() > b.contours.size();});
	//for (auto& reg: outReg) std::cout << reg.contours.size() << " ";
	//std::cout << std::endl;

	return ;
}

void regionStatistic(vector<vector<estimationItem>>&items, vector<tuple<int, int>>& region, areaInfo& info) {
	if (region.empty()) {
		std::cerr << "region to statistic is empty" << std::endl;
		return ;
	}
	int min_x = std::numeric_limits<int>::max(), max_x = std::numeric_limits<int>::min();
	int min_y = std::numeric_limits<int>::max(), max_y = std::numeric_limits<int>::min();
	int bcount = region.size();
	float totalDist = 0.0;
	float min_depth = std::numeric_limits<float>::max();
	float max_depth = std::numeric_limits<float>::min();
	info.contours.clear();
	for (int i=0; i < region.size(); i++) {
		int col = std::get<1>(region[i]);
		int row = std::get<0>(region[i]);
		if (items[row][col].mean == 0.0) continue;
		totalDist += items[row][col].mean;
		if (min_x > items[row][col].center_x) min_x = items[row][col].center_x;
		if (max_x < items[row][col].center_x) max_x = items[row][col].center_x;
		if (min_y > items[row][col].center_y) min_y = items[row][col].center_y;
		if (max_y < items[row][col].center_y) max_y = items[row][col].center_y;
		if (min_depth > items[row][col].min_d) min_depth = items[row][col].min_d;
		if (max_depth < items[row][col].max_d) max_depth = items[row][col].max_d;
		info.contours.emplace_back(std::make_tuple(items[row][col].center_x, items[row][col].center_y));
	}

	info.roi.x = min_x;
	info.roi.y = min_y;
	info.roi.width = max_x - min_x;
	info.roi.height = max_y - min_y;
	info.min_depth = min_depth;
	info.max_depth = max_depth;
	info.x = (min_x + max_x) / 2;
	info.y = (min_y + max_y) / 2;
	info.depth = totalDist / bcount;

	return ;
}

bool isSiblingArea(const areaInfo& area_a, const areaInfo& area_b) {
	// distance of area mean depth
	if (abs(area_a.depth - area_b.depth) > 0.1) return false;

	// area depth difference should in -0.03 ~ +0.03m
	if (abs(area_a.max_depth - area_b.max_depth) > 0.03 || abs(area_a.min_depth - area_b.min_depth) > 0.03) return false;

	//calculate distance between cneter points
	int x_dist = area_a.x - area_b.x;
	int y_dist = area_a.y - area_b.y; 

	// TODO: other match conditions

	return true;
}