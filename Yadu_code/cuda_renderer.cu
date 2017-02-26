//test
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cuda_renderer.h"
#include "image.h"

struct globals_const {

    SceneName sceneName;

    int 	numCircles;
    float* 	position;
    float* 	velocity;
    float* 	color;
    float* 	radius;

    int 	imgWidth;
    int 	imgHeight;
    float* 	imgData;	
};

//constants for GPU to access
__constant__ globals_const cuConstParams;

//Clearing the image to initial snowflakes setting
__global__ void kernelClearImageSnowflake(){
	
	int image_X = blockIdx.x * blockDim.x + threadIdx.x;
	int image_Y = blockIdx.y * blockDim.y + threadIdx.y;

	int width 	= cuConstParams.imgWidth;
	int height 	= cuConstParams.imgHeight;
	
	if(image_X >= width || image_Y >= height){
		return;
	}
	
	int 	offset 	= 4* (image_Y * width + image_X);
	float	shader 	= 0.4f + 0.45f * static_cast<float>(height - image_Y) / height;
	float4 	value  	= make_float4(shader,shader,shader,1.f);
	
	//Writing it to GPU memory
	*(float4*)(&cuConstParams.imgData[offset]) = value;
}

Cuda_renderer::Cuda_renderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

//Allocating buffer memory to the image.
void Cuda_renderer::allocImageBuf(int width, int height{
		
		if(image){
			delete image;
		}
		image = new Image(width,height);
)

//Loading the scene
void Cuda_renderer::loadScene(SceneName scene){
	sceneName = scene;
	
	if(sceneName == SNOWFLAKES){
		//Write an algorithm
	}
}

static void genRandomCircle(  int 		numCircles,
							  float*	position,
							  float*	velocity,
							  float*	color,
							  float*	radius){

		srand(0);
		std::vector<float> depths()
								  
}

//Clearing image for the renderer
void Cuda_renderer::clearImage(){
	
	//256 threads per blockDim
	dim3 blockDim(16,16,1);
	dim3 gridDim( 
				(image->width  + blockDim.x - 1) / blockDim.x,
				(image->height + blockDim.y - 1) / blockDim.y
	);
	
	if(sceneName == SNOWFLAKES){
		kernelClearImageSnowflake<<gridDim, blockDim>>();
	}else{
		//KernelClearImage call
	}
	cudaThreadSynchronize();
}





















