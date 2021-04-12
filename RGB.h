#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>

using namespace cv;


template<typename T>
struct RGB {
	T red;
	T green;
	T blue;

	RGB() 
	{
		red = 0;
		green = 0;
		blue = 0;
	}

	RGB(T red, T green, T blue) 
	{
		this.red = red;
		this.green = green;
		this.blue = blue;
	}

	template<typename F>
	RGB<T>& operator=(const RGB<F>& source)
	{
		if ((void*)this != (void*)(&source)) {
			this->red = (T)source.red;
			this->green = (T)source.green;
			this->blue = (T)source.blue;
		}

		return *this;
	}
};
