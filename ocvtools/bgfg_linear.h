
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

namespace cv {

class BackgroundSubtractorLin : public BackgroundSubtractor
{
private:
	int historySize;
	Mat history;
	Mat diff;
	Mat diffGray;
	double rate;
	double treshold;
public:
	BackgroundSubtractorLin(int history = 100, double treshold = 50) {
		historySize = history;
		rate = 1.0/history;
		this->treshold = treshold;
	}

	// takes the next video frame and returns the current foreground mask as 8-bit binary image.
	void apply(InputArray image, OutputArray fgmask, double x=-1) {
		Mat img = image.getMat();

		if (history.empty()) {
			img.convertTo(history, img.type());
		}

		// compute fg
		absdiff(history, img, diff);
		cvtColor(diff, diffGray, COLOR_BGR2GRAY);
		compare(diffGray, Scalar(treshold), fgmask, CV_CMP_GT);

		// update history
		addWeighted(history, (1-rate), img, rate, 0, history);
	}

	//! computes a background image
	void getBackgroundImage(OutputArray backgroundImage) const {
	}
};

}
