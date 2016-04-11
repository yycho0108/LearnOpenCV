#include "PoolLayer.h"

PoolLayer::PoolLayer(Size s_p, Size s_s):s_p(s_p),s_s(s_s){

}

std::vector<Mat>& PoolLayer::FF(std::vector<Mat> _I){
	s_i = _I[0].size();
	S.resize(_I.size());
	O.resize(_I.size());

	I.swap(_I);
	auto pw = s_p.width;
	auto ph = s_p.height;
	auto sw = s_s.width;
	auto sh = s_s.height;
	
	auto iw = I[0].size().width;
	auto ih = I[0].size().height;
	
	auto it_w = (iw - pw + sw-1) / sw;
	auto it_h = (ih - ph + sh-1) / sh;


	double maxVal=0;
	int maxIdx[2]={0,0};

	for(size_t i=0;i<I.size();++i){
		S[i].resize(it_h);
		O[i] = Mat(it_h,it_w,DataType<float>::type);
		for(int _y=0;_y<it_h;++_y){
			S[i][_y].resize(it_w);
			for(int _x=0;_x<it_w;++_x){
				auto y = _y*sh;
				auto x = _x*sw;

				auto _ph = min(ph,ih-y);
				auto _pw = min(pw,iw-x);
				/*maxVal=0;
				maxIdx[0]=maxIdx[1]=0;
				
				for(int yi=0;yi<_ph;++yi){
					for(int xi=0;xi<_pw;++xi){
						val = I[i].at<float>(y+yi,x+xi);
						if(val > maxVal){
							maxVal = val;
							maxIdx[1] = yi;
							maxIdx[0] = xi;
						}

					}
				}*/
				cv::minMaxIdx(I[i](Rect(x,y,_pw,_ph)),nullptr,&maxVal,nullptr,maxIdx);
				S[i][_y][_x] = Point(maxIdx[1],maxIdx[0]);
				O[i].at<float>(_y,_x) = maxVal;

			}

		}
	}
	
	return O;
}
std::vector<Mat>& PoolLayer::BP(std::vector<Mat> _G){
	G.resize(_G.size()); // = resize depth (channel)

	int h = S[0].size();
	int w = S[0][0].size();

	auto sw = s_s.width;
	auto sh = s_s.height;
	//cout << "_G[0] : " << endl << _G[0] << endl;
	for(size_t i=0;i<_G.size();++i){
		G[i] = Mat::zeros(I[i].size(),DataType<float>::type);
		for(int _y=0;_y<h;++_y){
			for(int _x=0;_x<w;++_x){
				auto& loc = S[i][_y][_x];
				auto val = _G[i].at<float>(_y,_x);
				G[i].at<float>(_y*sh +loc.y, _x*sw + loc.x) = val;
			}
		}
	}
	//cout << "G[0] : " << endl << G[0] << endl;
	return G;
}

void PoolLayer::setup(Size s){
	s_i = s;
}
Size PoolLayer::outputSize(){
	int w = (s_i.width-s_p.width+s_s.width-1)/s_s.width;
	int h = (s_i.height-s_p.height+s_s.height-1)/s_s.height;
	return Size(w,h);
}

void PoolLayer::save(FileStorage&, int){

}

void PoolLayer::load(FileStorage&, int){

}

