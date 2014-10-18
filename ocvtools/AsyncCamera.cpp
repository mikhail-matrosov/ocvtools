#pragma once

#include "AsyncCamera.h"
#include <unistd.h>
#include <sys/prctl.h>

using namespace cv;

Mat* AsyncCamera::smallocMat(int w, int h) {
	uchar *data = (uchar*) mmap(NULL, 4*w*h, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	Mat *m = (Mat*) mmap(NULL, sizeof(Mat), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	Mat *tmp = new Mat(h, w, CV_8UC3, data);
	memcpy(m, tmp, sizeof(Mat));

	return m;
}

long long getTimeNs() {
	timespec ts;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
	return ts.tv_sec*1000000000L + ts.tv_nsec;
}

void AsyncCamera::initAsyncVideoCapture(int cam, int w, int h) {
	// allocate shared memory
	retrieveId = (int*) mmap(NULL, sizeof(int*), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	retrieveTimestamps = (long long*) mmap(NULL, N_BUFFERS*sizeof(long long*), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	*retrieveId = 0;
	*retrieveTimestamps = 0;

	// init Matrices
	for (int i=0; i<N_BUFFERS; i++) {
		matrices[i] = smallocMat(w, h);
	}

	// create child process
	int childPID = fork();
    if (childPID) { // parent
    	printf("Created child process %d\n", childPID);

    } else { // child
    	// specify that the children should die right after parent.
    	prctl(PR_SET_PDEATHSIG, SIGHUP);

    	VideoCapture capture(cam);
    	capture.set(CV_CAP_PROP_FRAME_WIDTH, w);
    	capture.set(CV_CAP_PROP_FRAME_HEIGHT, h);
    	capture.set(CV_CAP_PROP_TEMPERATURE, 5000);

    	while(true) { //retrieveLoop
    		int ix = (*retrieveId+1)%N_BUFFERS;
    		Mat *m = matrices[ix];

    		uchar *data = m->data;
    		capture.read(*m);

    		// check if data was redirected
    		if (data != m->data) {
    			memcpy(data, m->data, m->dataend-m->datastart);
    			m->data = data;
    			m->datalimit = data + (m->dataend-m->datastart);
    			m->datastart = m->data;
    			m->dataend = m->datalimit;
    		}

    		retrieveTimestamps[ix] = getTimeNs();
    		(*retrieveId)++;
    	}
    }
}

// note that you should copy returned mat
Mat* AsyncCamera::get(long long &timestamp) {
	while (lastRetrievedId == *retrieveId) usleep(0);

	lastRetrievedId = *retrieveId;
	int ix = lastRetrievedId % N_BUFFERS;

	timestamp = retrieveTimestamps[ix];
	return matrices[ix];
}
Mat* AsyncCamera::get() {
	while (lastRetrievedId == *retrieveId) usleep(0);
	lastRetrievedId = *retrieveId;
	return matrices[lastRetrievedId % N_BUFFERS];
}

AsyncCamera::AsyncCamera() {
	initAsyncVideoCapture(0, 640, 480);
}
AsyncCamera::AsyncCamera(int cam=0) {
	initAsyncVideoCapture(cam, 640, 480);
}
AsyncCamera::AsyncCamera(int w, int h) {
	initAsyncVideoCapture(0, w, h);
}
AsyncCamera::AsyncCamera(int cam, int w, int h) {
	initAsyncVideoCapture(cam, w, h);
}
