#ifndef __LAYER_H__
#define __LAYER_H__
#include "Utility.h"
#include <fstream>

class Layer{
	public:
		virtual ~Layer(){};

		virtual std::vector<Mat>& FF(std::vector<Mat>)=0;
		virtual std::vector<Mat>& BP(std::vector<Mat>)=0;
		
		virtual void update(){};
		virtual void setup(Size){};
		virtual Size outputSize()=0;

		virtual void save(FileStorage& f, int i)=0;
		virtual void load(FileStorage& f, int i)=0;
};

#endif
