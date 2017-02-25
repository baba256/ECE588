struct Image{
	
	int 	width;
	int 	height;
	float* 	data;
	
	Image(int w,int h){
			width 	= w;
			height 	= h;
			data 	= new float[4 * width * height];
	}
	
	void clear(float r, float g, float b, float a){
		 int numPixels = width * height;
		 
	}
}