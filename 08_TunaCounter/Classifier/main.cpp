/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/photo/photo.hpp>

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame, CascadeClassifier& clf)
{
	std::vector<Rect> objs;
	Mat frame_gray;

	fastNlMeansDenoisingColored(frame,frame);
    //cvtColor(frame,frame,COLOR_BGR2HSV);
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY );
	imshow("THRESH", frame_gray);
	equalizeHist( frame_gray, frame_gray );

	//-- Detect objectss
	clf.detectMultiScale( frame_gray, objs, 1.05, 5, 0, Size(8,8), Size(96,96));

	for( size_t i = 0; i < objs.size(); i++ )
	{
		//-- Draw the face
		Point center( objs[i].x + objs[i].width/2, objs[i].y + objs[i].height/2 );
		ellipse( frame, center, Size( objs[i].width/2, objs[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
	}
	//-- Show what you got
	imshow("FRAME", frame );
}

/**
 * @function main
 */
int main(int argc, char* argv[])
{
	if(argc != 3){
		fprintf(stderr, "USAGE : %s <img> <clf>", argv[0]);
		return -1;
	}

	CascadeClassifier tuna_cascade;

	Mat img = imread(argv[1]);
	resize(img,img,Size(0,0),0.2,0.2);
	tuna_cascade.load(argv[2]);

	namedWindow("FRAME");
	imshow("FRAME", img);
	waitKey();
	//-- 3. Apply the classifier to the frame
	detectAndDisplay(img, tuna_cascade);

	//-- bail out if escape was pressed
	int c;
	while((c = waitKey(10))){
		if(c == 27)
			break;
		else if(c == 'r'){
			//rerun with different image
		}
	}

	return 0;
}


