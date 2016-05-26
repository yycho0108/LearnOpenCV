#include <opencv2/core/core.hpp>
#include <iostream>


int main(){
	char data[4] = {1,2,3,4};
	cv::Mat m(2,2,cv::DataType<char>::type, data);
	m.convertTo(m,cv::DataType<float>::type);
	m += 1;
	for(int i=0;i<4;++i){
		std::cout << (int)data[i] << ' ';
	}
	std::cout << m;
}
