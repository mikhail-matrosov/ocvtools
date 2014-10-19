#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

class LKSmooth
{
public:
	LKSmooth();
	void apply(Mat &output, vector<Point2f> &points, vector<Point2f> &vels);
	void visualizeVField(Mat &src, Mat &dst);

private:
	Mat norma;
	Mat norma2;
};
