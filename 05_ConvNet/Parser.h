#ifndef __PARSER_H__
#define __PARSER_H__

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class Parser{

private:
	ifstream f_d;
	ifstream f_l;
	unsigned char buf_d[28*28];
	unsigned char buf_l[1];
public:
	Parser(string d, string l);
	bool read(Mat& d, Mat& l);
	void reset();
	~Parser();
};

#endif
