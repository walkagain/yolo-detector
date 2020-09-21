#ifndef     __PROFILE_DETECT_H_
#define     __PROFILE_DETECT_H_
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

class ProfileDetHelper {
public:
    ProfileDetHelper();
    void ProfileDetect(cv::Mat& img, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy);
    void ProfileDetect(cv::Mat& img, int x, int y, int w, int h, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy);
    void ContoursTopN(vector<vector<cv::Point> >& contours, vector<int>& indexs, int topN = 5);
    void ContoursDraw(cv::Mat& img, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy);
    void ContoursDrawByIndexs(cv::Mat& img, vector<vector<cv::Point> >& contours, const vector<int>& indexs);
};
#endif  //__PROFILE_DETECT_H_