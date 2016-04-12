#include "Parser.h"

Parser::Parser(string d, string l){
	f_d.open(d);
	f_l.open(l);
	reset();
}

bool Parser::read(Mat& d, Mat& l){
	//std::cout << "READING " << std::endl;

	f_d.read((char*)buf_d,28*28);
	f_l.read((char*)buf_l,1);

	d = Mat(28,28,DataType<unsigned char>::type,buf_d);
	d.convertTo(d,DataType<float>::type,1/256.0);

	//d *= 1.275;
	//d -= 0.1;
	
	//will normalize input as described in LeNet paper. let's see how it goes

	l = Mat::zeros(10,1,DataType<float>::type);
	l.at<float>(buf_l[0],0) = 1.0;
	
	//std::cout << "READING OVER" << std::endl;
	return f_d && f_l;
}

void Parser::reset(){
	f_d.clear();
	f_d.seekg(16,ios::beg);
	f_l.clear();
	f_l.seekg(8,ios::beg);
}

Parser::~Parser(){
	f_d.close();
	f_l.close();
}

