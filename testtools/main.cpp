#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "bgfg_linear.h"

#include "LKTracker.cpp"
#include "LKSmooth.h"

using namespace cv;
using namespace std;

// returns milliseconds
long long time_ms() {
	struct timespec ts;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
	return ts.tv_sec*1000 + ts.tv_nsec/1000000;
}

double framerate=1;
long long prevT = 0;
double updFramerate() {
	long long t = time_ms();
	double framerateNew = 1000.0/(t-prevT);
	if (framerateNew>1.5*framerate || framerateNew*1.5<framerate) {
		framerate = framerateNew;
	} else {
		framerate = framerate*0.9 + framerateNew*0.1;
	}
	prevT = t;
	return framerate;
}

int main(int argc, char** argv) {
	BackgroundSubtractorLin *bgs = new BackgroundSubtractorLin(100, 50);

	VideoCapture cap(1); // open the default camera
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if(!cap.isOpened())  // check if we succeeded
		return -1;

	Mat frame;
	Mat frameFG;
	Mat small;
	Mat edges;
	Mat gray, prevGray;
	Mat segmask;
	Mat markers(480, 640, CV_32SC1);
	vector<Rect> boundingRects;
	Mat fground;
	Mat fgroundPrev;
	Mat lkField(Size(1,1), CV_32FC2);

    //Size winSize(31,31);

    LKTracker lkt;
    LKSmooth lks;

	for(;;)
	{
		cap >> frame;

		if (prevGray.empty()) {
			cvtColor(frame, gray, COLOR_BGR2GRAY);
			gray.copyTo(prevGray);
		} else {
			gray.copyTo(prevGray);
			cvtColor(frame, gray, COLOR_BGR2GRAY);
		}

		resize(lkField, lkField, frame.size());

		fground.copyTo(fgroundPrev);

	    //update the background model
	    bgs->apply(frame, fground);

	    medianBlur(fground, fground, 3);

	    if (fgroundPrev.empty())
	    	fground.copyTo(fgroundPrev);
		lkt.apply(prevGray, gray, fgroundPrev);

		vector<Point2f> *points = lkt.getPoints();
		vector<Point2f> *vels = lkt.getVelocities();

		cvtColor(fground, frameFG, COLOR_GRAY2RGB);
		addWeighted(frame, 0.5, frameFG, 0.5, 0, frame);

        for( int i=0; i < points->size(); i++ ) {
        	Point2f p = points->at(i);
        	Point2f v = vels->at(i);
        	circle(frame, p, 3, Scalar(0,255,0), 2);
        	line(frame, p-v, p, Scalar(0,255,0), 2);
        }

	    resize(frame, small, Size(0, 0), 0.5, 0.5);

		//imshow("xor", gray);
        char buffer[64];
        sprintf(buffer, "%dx%d, %.1f FPS", frame.cols, frame.rows, updFramerate());
        cv::putText(frame, string(buffer), Point(10, 30), 1, 1, Scalar(0,255,0));

        lks.apply(lkField, *points, *vels);
        lks.visualizeVField(lkField, edges);

        add(frameFG, Scalar(200, 200, 200), frameFG);
        multiply(edges, frameFG, edges, 1.0/256);

        imshow("lk", edges);
		imshow("frame", frame);
		if(waitKey(1) >= 0) break;

		//printf("%dx%d, %.2f FPS\n", frame.cols, frame.rows, updFramerate());
	}

	return 0;
}
