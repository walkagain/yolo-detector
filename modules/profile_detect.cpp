#include "profile_detect.h"

ProfileDetHelper::ProfileDetHelper() {

}

void ProfileDetHelper::ProfileDetect(cv::Mat& img, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy) {
    contours.clear();
    hierarchy.clear();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::GaussianBlur(gray, binary,cv::Size(3,3),0);
	cv::Canny(binary, binary, 100, 255, 3, true);
    //cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    cv::findContours(binary, contours, hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    return ;
}

void ProfileDetHelper::ProfileDetect(cv::Mat& img, int x, int y, int w, int h, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy) {
    contours.clear();
    hierarchy.clear();
    cv::Mat bbox(img, cv::Rect(x,y, w, h));
    cv::Mat box_img = bbox.clone();
    //printf("rows:%d, columns:%d, dims:%d\n", box_img.rows, box_img.cols, box_img.dims);
    ProfileDetect(box_img, contours, hierarchy);
    
    //TODO: adjust the contours to original image
    for (int i=0; i < contours.size(); i++) {
        for (int j=0; j < contours[i].size(); j++) {
            contours[i][j].x += x;
            contours[i][j].y += y;
        }
    }
    return ;
}

void ProfileDetHelper::ContoursTopN(vector<vector<cv::Point> >& contours, vector<int>& indexs, int topN) {
    vector<double> areas;
    indexs.clear();
    for (int i = 0; i < contours.size(); i++) {
        areas.emplace_back(cv::contourArea(contours[i]));
    }
    
    double mArea = 0.0, idx = -1;
    topN = topN < contours.size()?topN:contours.size();
    for (int i=0; i < topN; i++) {
        for (int j=0; j < areas.size(); j++) {
            if (mArea < areas[i]) {
                mArea = areas[j];
                idx = i;
            }
        }
        if (idx == -1) continue;
        areas[idx] = -1.0;
        mArea = 0.0;
        idx = -1;
        indexs.emplace_back(idx);
    }
    sort(indexs.begin(), indexs.end());
    return ;
}

void ProfileDetHelper::ContoursDraw(cv::Mat& img, vector<vector<cv::Point> >& contours, vector<cv::Vec4i>& hierarchy) {
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        //if (area < 100.0) continue;
        cv::drawContours(img, contours, i, cv::Scalar(255, 0, 0), cv::FILLED, 8, hierarchy, 0);
    }
    return ;
}

void ProfileDetHelper::ContoursDrawByIndexs(cv::Mat& img, vector<vector<cv::Point> >& contours, const vector<int>& indexs) {
    if (indexs.empty()) return ;
    int idx = 0;

    for (int i = 0; i < contours.size(); i++) {
        if (idx >= indexs.size() || indexs[idx] != i) continue;
        idx ++;
        cv::drawContours(img, contours, i, cv::Scalar(255, 0, 0), cv::FILLED, 8, vector<cv::Vec4i>(), 0);
    }
    return ;
}