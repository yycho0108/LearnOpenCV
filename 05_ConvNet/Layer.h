#ifndef __LAYER_H__
#define __LAYER_H__
#include "Utility.h"

class Layer{
	public:
		virtual ~Layer(){};

		virtual std::vector<Mat>& FF(std::vector<Mat>)=0;
		virtual std::vector<Mat>& BP(std::vector<Mat>)=0;
		
		virtual void update(){};
		virtual void setup(Size){};
		virtual Size outputSize()=0;
};

#endif
