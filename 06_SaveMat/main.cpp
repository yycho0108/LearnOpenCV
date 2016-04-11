#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <cstdio>

using namespace cv;

Mat load(std::string f_in, std::string name){
	Mat m;
	FileStorage fs(f_in, FileStorage::READ);
	fs[name] >> m;
	return m;
}

void save(std::string f_out, std::string name, Mat& m){
	FileStorage fs(f_out, FileStorage::APPEND);
	fs << name << m;
}

int main(){
	Mat A = (Mat_<float>(2,5) << 1,3,2,4,6,7,5,8,9,0);
	Mat B = (Mat_<float>(5,2) << 10,3,22,4,6,7,5,8,9,0);
	Mat C = (Mat_<float>(3,3) << 11,3,23,4,6,7,8,9,0);
	std::remove("save");
	save("save","A",A);
	save("save","B",B);
	save("save","C",C);
	//Mat A = load("save","A");
	//std::cout << "A : " << A << std::endl;
	
}
