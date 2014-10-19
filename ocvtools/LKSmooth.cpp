#pragma once

#include "LKSmooth.h"

LKSmooth::LKSmooth() {}

void LKSmooth::apply(Mat &output, vector<Point2f> &points, vector<Point2f> &vels) {
	//resize(output, output, Size(0,0), 0.25, 0.25);
	output.convertTo(output, CV_32FC2);
	if (norma.empty()) {
		norma = Mat(output.size(), CV_32FC1);
		norma2 = Mat(output.size(), CV_32FC2);
	} else {
		if (output.rows != norma.rows || output.cols != norma.cols) {
			resize(norma, norma, output.size());
			resize(norma2, norma2, output.size());
		}
	}
	norma.setTo(Scalar(1e-9f));

	output.setTo(Scalar(.0f));

	Size blurSize(7,7);

    for( int i=0; i < points.size(); i++ ) {
    	Point2f p = points.at(i);
    	Point2f v = vels.at(i);
    	int x = (int)p.x;
    	int y = (int)p.y;
    	x = x<0?0: (x>output.cols ? output.cols : x);
    	y = y<0?0: (y>output.rows ? output.rows : y);
    	output.at<Point2f>(y, x) = v;
    	norma.at<float>(y, x) = 1.0;
    }

    GaussianBlur(output, output, Size(51,51), 5, 5);
    GaussianBlur(norma, norma, Size(51,51), 5, 5);

	vector<Mat> channels(2);
	channels[0] = norma;
	channels[1] = norma;
	merge(channels, norma2);

    divide(output, norma2, output);

	//resize(output, output, Size(0,0), 4, 4);
}

void LKSmooth::visualizeVField(Mat &src, Mat &dst) {
	vector<Mat> channels(src.dims);
	vector<Mat> channels8(3);
	split(src, channels);

	for(int i=0; i<min(src.dims,3); i++) {
		channels[i].convertTo(channels8[i], CV_8UC1, 1, 127);
	}

	for(int i=src.dims; i < 3; i++) {
		channels8[i] = Mat(src.size(), CV_8UC1, Scalar(127));
	}

	//for(int i=0; i<3; i++)
	//	cout << channels[i].type() << channels[i].rows << channels[i].cols << endl;
	//cout << endl;

	merge(channels8, dst);
}

