#include "render_circle.h"

class Cuda_renderer : public render_circle{
	
private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

public:
		Cuda_renderer();
		
		void image_setup();
		
		void allocImageBuf(int width, int height);
		
		void clearImage();
		
		void render();
		
		void pixel_shader( int Index,
						   float pixelx, float pixely,
						   float p_x,float p_y, float p_z,
						   float* pixelData);
}
