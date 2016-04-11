#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <cstdio>

using namespace cv;

Mat load(FileStorage& fs, std::string name){
	Mat m;
	fs[name] >> m;
	return m;
}

void save(FileStorage& fs, std::string name, Mat& m){
	fs << name << m;
}

void testLoad(){
	FileStorage fs("save.yml", FileStorage::READ);
	Mat A = load(fs,"A");
	std::cout << "A : " << A << std::endl;
	Mat B = load(fs,"B");
	std::cout << "B : " << B << std::endl;
	Mat C = load(fs,"C");
	std::cout << "C : " << C << std::endl;
	
}

void testSave(){
	std::remove("save.yml");
	FileStorage fs("save.yml", FileStorage::WRITE);
	Mat A = (Mat_<float>(2,5) << 1,3,2,4,6,7,5,8,9,0);
	Mat B = (Mat_<float>(5,2) << 10,3,22,4,6,7,5,8,9,0);
	Mat C = (Mat_<float>(3,3) << 11,3,23,4,6,7,8,9,0);
	save(fs,"A",A);
	save(fs,"B",B);
	save(fs,"C",C);
}

int main(){
	//testSave();
	testLoad();

}
