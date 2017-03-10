#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "vector_types.h"
#include "functional"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "util.h"
#include "cuda_renderer.h"
#include "image.h"

#define cudaCheckError() { \
		cudaError_t e=cudaGetLastError(); \
		if(e!=cudaSuccess) { \
			printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
			exit(0); \
		} \
}
//#include "exclusiveScan.cu_inl"

// Threads Per Block
#define TPB_X 16
#define TPB_Y 16
#define TPB (TPB_X*TPB_Y)

// Pixels Per Thread
#define PPT_X 2
#define PPT_Y 2
#define PPT (PPT_X * PPT_Y)

// Pixels Per Block
#define PPB_X (PPT_X * TPB_X)
#define PPB_Y (PPT_Y * TPB_Y)
#define PPB (PPB_X * PPB_Y)

// Circle Per Thread
#define CIRCLES_PER_THREAD 32

// Total circles affects certain region
#define TOTAL 3500

// randomFloat --
// //
// // return a random floating point value between 0 and 1
static float randomFloat() {
	return static_cast<float>(rand()) / RAND_MAX;
}


struct globals_const{
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
//__global__ void kernelClearImageSnowflake(){
//
//	int image_X = blockIdx.x * blockDim.x + threadIdx.x;
//	int image_Y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	int width 	= cuConstParams.imgWidth;
//	int height 	= cuConstParams.imgHeight;
//
//	if(image_X >= width || image_Y >= height){
//		return;
//	}
//
//	int 	offset 	= 4* (image_Y * width + image_X);
//	float	shader 	= 0.4f + 0.45f * static_cast<float>(height - image_Y) / height;
//	float4 	value  	= make_float4(shader,shader,shader,1.f);
//
//	//Writing it to GPU memory
//	*(float4*)(&cuConstParams.imgData[offset]) = value;
//}


/*
__device__ __inline__ int circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - 1.2*circleRadius) &&
         circleX <= (boxR + 1.2*circleRadius) &&
         circleY >= (boxB - 1.2*circleRadius) &&
         circleY <= (boxT + 1.2*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

 */
/*
__inline__ __device__ void sharedMemExclusiveScan(int threadIndex, uint* sInput, uint* sOutput, volatile uint* sScratch, uint size)
{
    if (size > WARP_SIZE) {

        uint idata = sInput[threadIndex];

        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);

        // Save top elements of each warp for exclusive warp scan sync
        // to wait for warp scans to complete (because s_Data is being
        // overwritten)
        __syncthreads();

        if ( (threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;

        // wait for warp scans to complete
        __syncthreads();

        if ( threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
            // grab top warp elements
            uint val = sScratch[threadIndex];
            // calculate exclusive scan and write back to shared memory
            sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();

        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE] - idata;

    } else if (threadIndex < WARP_SIZE) {
        uint idata = sInput[threadIndex];
        sOutput[threadIndex] = warpScanExclusive(threadIndex, idata, sScratch, size);
    }
}
 */
__device__ __inline__ void pixel_shader(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

	float diffX = p.x - pixelCenter.x;
	float diffY = p.y - pixelCenter.y;
	float pixelDist = diffX * diffX + diffY * diffY;

	float rad = cuConstParams.radius[circleIndex];;
	float maxDist = rad * rad;

	// circle does not contribute to the image
	if (pixelDist > maxDist)
		return;

	float3 rgb;
	float alpha;

	// there is a non-zero contribution.  Now compute the shading value

	// simple: each circle has an assigned color
	int index3 = 3 * circleIndex;
	rgb = *(float3*)&(cuConstParams.color[index3]);
	alpha = .5f;

	float oneMinusAlpha = 1.f - alpha;

	float4 existingColor = *imagePtr;
	float4 newColor;
	newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
	newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
	newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
	newColor.w = alpha + existingColor.w;

	// global memory write
	*imagePtr = newColor;
}

__global__ void kernelClearImage(float r, float g, float b, float a) {

	int image_X = blockIdx.x * blockDim.x + threadIdx.x;
	int image_Y = blockIdx.y * blockDim.y + threadIdx.y;

	int width 	= cuConstParams.imgWidth;
	int height 	= cuConstParams.imgHeight;

	if (image_X >= width || image_Y >= height)
		return;

	int offset = 4 * (image_Y * width + image_X);
	float4 value = make_float4(r, g, b, a);

	//Writing it to GPU memory
	*(float4*)(&cuConstParams.imgData[offset]) = value;
}


#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"

__global__ void kernelRenderCircles(){

//	printf( "Render kernel start \n");
	int threadIndex = threadIdx.y * TPB_X + threadIdx.x;
	__shared__ unsigned int circleOrder[TOTAL];
	__shared__ unsigned int circleCount[TPB];
    __shared__ unsigned int circleIndex[TPB];

	short imageWidth 	= cuConstParams.imgWidth;
    short imageHeight 	= cuConstParams.imgHeight;
    float invWidth 		= 1.f / imageWidth;
    float invHeight 	= 1.f / imageHeight;

	//Computing Box for region
	short regionMinX = PPB_X * blockIdx.x;
    short regionMaxX = PPB_X * (blockIdx.x + 1) - 1;
    short regionMinY = PPB_Y * blockIdx.y;
    short regionMaxY = PPB_Y * (blockIdx.y + 1) - 1;

	//Normalizing
	float boxL = invWidth * regionMinX;
    float boxR = invWidth * regionMaxX;
    float boxB = invHeight * regionMinY;
    float boxT = invHeight * regionMaxY;

	//Finding the parameters of circles that affect the region
	int numCircles 		 = cuConstParams.numCircles;
	int circlesPerThread = (numCircles + TPB - 1) / TPB;
	int circleStart		 = threadIndex * circlesPerThread;
	int circleEnd		 = circleStart + (circlesPerThread - 1);

	if(threadIndex == TPB - 1){
		circleEnd = numCircles - 1;
	}
//	printf( "Render kernel line 240 \n");
	//Allocating private thread
	unsigned int privateCircleOrder[CIRCLES_PER_THREAD];
	int privateCircleCount = 0;

	//Counting the circles in region
	for(int i= circleStart; i<= circleEnd; i++){
		int index3 = 3 * i;
		//Current position and radius
		float3 p = *(float3*)(&cuConstParams.position[index3]);
		float  rad = cuConstParams.radius[i];
        if( circleInBoxConservative(p.x, p.y, rad, boxL, boxR, boxT, boxB) )
            privateCircleOrder[privateCircleCount++] = i;
	}
//	printf( "Render kernel line 254 \n");
	//Total Final count has to be stored in Index
	circleCount[threadIndex] = privateCircleCount;
	//Syncing the Threads
	__syncthreads();

//	printf( "Render kernel line b4 shared mem scan \n");

	//Performing scanning on circle Index
	 sharedMemExclusiveScan(threadIndex, circleCount, circleIndex, circleOrder, TPB);
    __syncthreads();

  //  printf( "Render kernel line after shared mem scan \n");

	// Use circleIndex array to store privateCircleOrder
    int total =  circleCount[TPB-1] + circleIndex[TPB-1];
    int privateIndex = circleIndex[threadIndex];

	for(int i = 0; i < privateCircleCount; i++) {
        circleOrder[privateIndex++] = privateCircleOrder[i];
    }
    __syncthreads();

	//Rendering the pixel in the region
	//FIXME: Check this logic
    for(int i = 0; i < total; i++) {
        int index = circleOrder[i];
        int index3 = 3 * index;

        // Read position
        float3 p = *(float3*)(&cuConstParams.position[index3]);

        //for each pixel in this thread of this block
        for (int pindex = 0; pindex < PPT; pindex++) {
            int pixelIndex = threadIndex + pindex * TPB;
            int pixelX = regionMinX + pixelIndex % PPB_X;
            int pixelY = regionMinY + pixelIndex / PPB_X;
            // read info of pixel
            float4* imgPtr = (float4*)(&cuConstParams.imgData[4 * (pixelY * imageWidth + pixelX)]);
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            pixel_shader(index, pixelCenterNorm, p, imgPtr);
        }

    }


}

__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

	float diffX = p.x - pixelCenter.x;
		float diffY = p.y - pixelCenter.y;
		float pixelDist = diffX * diffX + diffY * diffY;

		float rad = cuConstParams.radius[circleIndex];;
		float maxDist = rad * rad;

		// circle does not contribute to the image
		if (pixelDist > maxDist)
			return;

		float3 rgb;
		float alpha;

		// there is a non-zero contribution.  Now compute the shading value

		// simple: each circle has an assigned color
		int index3 = 3 * circleIndex;
		rgb = *(float3*)&(cuConstParams.color[index3]);
		alpha = .5f;

		float oneMinusAlpha = 1.f - alpha;

		float4 existingColor = *imagePtr;
		float4 newColor;
		newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
		newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
		newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
		newColor.w = alpha + existingColor.w;

		// global memory write
		*imagePtr = newColor;

	// END SHOULD-BE-ATOMIC REGION
}

/*
__global__ void kernelRenderCircles() {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= cuConstParams.numCircles)
		return;

	int index3 = 3 * index;

	// read position and radius
	float3 p = *(float3*)(&cuConstParams.position[index3]);
	float  rad = cuConstParams.radius[index];

	// compute the bounding box of the circle. The bound is in integer
	// screen coordinates, so it's clamped to the edges of the screen.
	short imageWidth = cuConstParams.imgWidth;
	short imageHeight = cuConstParams.imgHeight;
	short minX = static_cast<short>(imageWidth * (p.x - rad));
	short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
	short minY = static_cast<short>(imageHeight * (p.y - rad));
	short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

	// a bunch of clamps.  Is there a CUDA built-in for this?
	short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
	short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
	short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
	short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

	float invWidth = 1.f / imageWidth;
	float invHeight = 1.f / imageHeight;

	// for all pixels in the bonding box
	for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
		float4* imgPtr = (float4*)(&cuConstParams.imgData[4 * (pixelY * imageWidth + screenMinX)]);
		for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
			float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
					invHeight * (static_cast<float>(pixelY) + 0.5f));
			shadePixel(index, pixelCenterNorm, p, imgPtr);
			imgPtr++;
		}
	}
}
*/

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

Cuda_renderer::~Cuda_renderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}
const Image*
Cuda_renderer::image_setup() {  
	printf("Copying image data from device\n");

	cudaMemcpy(image->data,
			cudaDeviceImageData,
			sizeof(float) * 4 * image->width * image->height,
			cudaMemcpyDeviceToHost);

	return image;
}

//Allocating buffer memory to the image.
void Cuda_renderer::allocImageBuf(int width, int height){

	if(image){
		delete image;
	}
	image = new Image(width,height);
}

void Cuda_renderer::render() {
	// 256 threads per block is a healthy number
	dim3 blockDim(TPB_X, TPB_Y, 1);
	dim3 gridDim(
			(image->width + PPB_X - 1) / PPB_X,
			(image->height + PPB_Y - 1) / PPB_Y);

    // 256 threads per block is a healthy number
   // dim3 blockDim(256, 1);
  //  dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);


	printf ("b4 done bro \n");
	kernelRenderCircles<<<gridDim, blockDim>>>();
	printf ("done bro \n");
	cudaCheckError();
	printf ("double done bro \n");
cudaDeviceSynchronize();
	printf ("quad done bro \n");
}

void Cuda_renderer::setup(){

	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	printf("Initializing CUDA for CudaRenderer\n");
	printf("Found %d Cuda device(s)\n",deviceCount);

	for(int i=0;i<deviceCount;i++){
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties (&deviceProps,i);
		printf("Device %d: %s\n", i, deviceProps.name);
		printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
		printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
	}

	cudaMalloc(&cudaDevicePosition	, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceVelocity	, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceColor		, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceRadius	, sizeof(float) * numCircles);
	cudaMalloc(&cudaDeviceImageData	, sizeof(float) * 4 * image->width * image->height);

	cudaMemcpy(cudaDevicePosition , position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceVelocity , velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceColor	  , color	, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceRadius   , radius	, sizeof(float) * numCircles	, cudaMemcpyHostToDevice);

	globals_const params;
	params.sceneName 	= sceneName;
	params.numCircles 	= numCircles;
	params.imgWidth 	= image->width;
	params.imgHeight 	= image->height;
	params.position 	= cudaDevicePosition;
	params.velocity 	= cudaDeviceVelocity;
	params.color 		= cudaDeviceColor;
	params.radius 		= cudaDeviceRadius;
	params.imgData 	= cudaDeviceImageData;

	cudaMemcpyToSymbol(cuConstParams, &params, sizeof(globals_const));

	printf ("setup done bro \n");
}


static void genRandomCircle(  int 		numCircles,
		float*	position,
		float*	velocity,
		float*	color,
		float*	radius){
	srand(0);
			std::vector<float> depths(numCircles);
			for(int i=0;i<numCircles;i++){

				float depth = depths[i];
				radius[i]	= 0.02f+ 0.06f * randomFloat();

				int index3 = 3*i;

				position[index3]  	= randomFloat();
				position[index3+1]	= randomFloat();
				position[index3+2]	= depth;

				if(numCircles <= 100){
					color[index3] 	= .1f + .9f * randomFloat();
					color[index3+1] = .2f + .5f * randomFloat();
					color[index3+2] = .5f + .5f * randomFloat();
				}else{
					color[index3] 	= .3f + .9f * randomFloat();
					color[index3+1] = .1f + .9f * randomFloat();
					color[index3+2] = .1f + .4f * randomFloat();
				}
			}

}

//Loading the scene
void Cuda_renderer::loadScene(SceneName scene){
	sceneName = scene;

	if(sceneName == SNOWFLAKES){
		//Write an algorithm
	} else if (sceneName == CIRCLE_Rand){
		numCircles 	= 10 * 100;

		position 	= new float[3 * numCircles];
		velocity	= new float[3 * numCircles];
		color		= new float[3 * numCircles];
		radius		= new float[numCircles];

		genRandomCircle(numCircles, position, velocity, color, radius);
		printf ("testing \n");
	} else {
		printf ("Error in loading the scene %s\n",sceneName);
	}
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
		//	kernelClearImageSnowflake<<<gridDim, blockDim>>>();
	}else{
		//KernelClearImage call

		kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
	}
cudaThreadSynchronize();

}
