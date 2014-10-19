#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef vector<Point2f> V;

class LKTracker {
private:
	int MAX_COUNT;
	int RANDOMIZED_LK;
	Size winSize;
	double pointsDensity;
	double motionTreshold2 = 1;
	int maxLevel;

	V * prevPoints = new V();
	V * nextPoints = new V();
	V * velocities = new V();

    vector<uchar> status;
    vector<float> err;

	void swapVectors() {
		V * p = prevPoints;
		prevPoints = nextPoints;
		nextPoints = p;
	}

	Point2i getRandomFrom(const Mat &nonZeros) {
		int n = nonZeros.total();
		int r = random() % n;
		Point2i p = nonZeros.at<Point2i>( random() % nonZeros.total() );
		return nonZeros.at<Point2i>( random() % nonZeros.total() );
	}

	void clamp(Point2f &p, int w, int h) {
		p.x = p.x >= w ? w-1 : (p.x < 0 ? 0:p.x);
		p.y = p.y >= h ? h-1 : (p.y < 0 ? 0:p.y);
	}

public:
	LKTracker(int max_points = 1000, int randomizedLK = 50,
			double pointsDensity = 0.002, int winSize = 15, int maxLevel = 8)
	{
		this->MAX_COUNT = max_points;
		this->RANDOMIZED_LK = randomizedLK;
		this->winSize = Size(winSize, winSize);
		this->pointsDensity = pointsDensity;
		this->maxLevel = maxLevel;
	}

	void apply(const Mat &prev, const Mat &next, const Mat &fgroundPrev) {
		Mat nonZeros;
		findNonZero(fgroundPrev, nonZeros);

		int w = next.cols;
		int h = next.rows;

		V * newNextPoints = new V();
		V * newPrevPoints = new V();

		// desired number of points
		int nDesired = pointsDensity*nonZeros.total();
		nDesired = min(nDesired, MAX_COUNT);
		int nExist = nextPoints->size();

		// choose good points
		for (int i=0; i<nExist; i++) {
			Point2f &pn = nextPoints->at(i);
			Point2f &pp = prevPoints->at(i);
			Point2f dd = pn-pp;

			// clamp pn coordinates
			clamp(pn, w, h);

			// if point is moving or on the white side
			if (dd.dot(dd) > motionTreshold2 || fgroundPrev.at<uchar>(pn))
			{
				newNextPoints->push_back(pn);
				newPrevPoints->push_back(pp);
			}
		}

		delete nextPoints;
		delete prevPoints;
		nextPoints = newNextPoints;
		prevPoints = newPrevPoints;

		if (nDesired < nExist) {
			nextPoints->resize(nDesired);
			prevPoints->resize(nDesired);
		}

		// randomize alive points
		int sz = nextPoints->size();
		for (int i=0; i<sz; i++) {
			Point2f &pn = nextPoints->at(i);
			if (random()%MAX_COUNT<RANDOMIZED_LK) {
				pn = getRandomFrom(nonZeros);
			}
		}

		// add points
		for (int i=nExist; i<nDesired; i++) {
			Point2f p = getRandomFrom(nonZeros);
			nextPoints->push_back(p);
			prevPoints->push_back(p);
		}

		swapVectors();

		if (prevPoints->size() > 0) {
			calcOpticalFlowPyrLK(prev, next,
					*prevPoints, *nextPoints, status, err, winSize, 5);
		}
	}

	V * getPoints() {
		return nextPoints;
	}

	V * getVelocities() {
		int psz = nextPoints->size();

		velocities->resize(psz);

		for (int i=0; i<psz; i++) {
			Point2f &v = velocities->at(i);
			v = nextPoints->at(i) - prevPoints->at(i);
		}

		return velocities;
	}
};
