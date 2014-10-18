#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

using namespace cv;

long long getTimeNs();

class AsyncCamera
{
public:
	AsyncCamera();
	AsyncCamera(int cam);
	AsyncCamera(int w, int h);
	AsyncCamera(int cam, int w, int h);

	// note that you should copy returned mat
	Mat* get();
	Mat* get(long long &timestamp);

private:
	int *retrieveId;
	int lastRetrievedId;
	long long *retrieveTimestamps;

	#define N_BUFFERS 4
	Mat* matrices[N_BUFFERS];

	Mat* smallocMat(int w, int h);
	void initAsyncVideoCapture(int cam, int w, int h);
};
