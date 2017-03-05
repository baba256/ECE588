#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "refRenderer.h"
#include "image.h"
#include "util.h"


// randomFloat --
// //
// // return a random floating point value between 0 and 1
static float
randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

RefRenderer::RefRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;
}

const Image* RefRenderer::image_setup() {
    return image;
}

void RefRenderer::setup() {
    //No setup required as we are running serially
}


void RefRenderer::allocImageBuf(int width, int height) {


    if (image)
        delete image;
    image = new Image(width, height);
}

void RefRenderer::clearImage() {

    // clear image to white
        image->clear(1.f, 1.f, 1.f, 1.f);
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
			radius[i]	= 0.2f+ 0.6f * randomFloat();
			
			int index3 = 3*i;
			
			position[index3]  	= randomFloat();
			position[index3+1]	= randomFloat();
			position[index3+2]	= depth;
			
			if(numCircles <= 1000){
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

void RefRenderer::loadScene(SceneName scene) {
	sceneName = scene;
	
	if(sceneName == SNOWFLAKES){
		//Write an algorithm
	} else if (sceneName == CIRCLE_Rand){

		numCircles 	= 11 * 100;

		
		position 	= new float[3 * numCircles];
		velocity	= new float[3 * numCircles];
		color		= new float[3 * numCircles];
		radius		= new float[numCircles];
		
		genRandomCircle(numCircles, position, velocity, color, radius);
	} else {
	//	printf (stderr,"Error in loading the scene %s\n",sceneName);
	}
}


//Shading the pixels

void RefRenderer::pixel_shader(int circleIndex, float pixelCenterX, float pixelCenterY, float px, float py, float pz, float* pixelData)

{
	float diffX = px - pixelCenterX;
    float diffY = py - pixelCenterY;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = radius[circleIndex];
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;
	
	float color_r, color_g, color_b, alpha;
	
	//Configuring Fixed-RGB colors
	int index3 = 3 * circleIndex;
    color_r = color[index3];
    color_g = color[index3+1];
    color_b = color[index3+2];
    alpha = .5f;
	
	pixelData[0] = alpha * color_r + alpha * pixelData[0];
    pixelData[1] = alpha * color_g + alpha * pixelData[1];
    pixelData[2] = alpha * color_b + alpha * pixelData[2];
    pixelData[3] += alpha;
}	
	
	
//Rendering the image
void RefRenderer::render() {

	  printf("start of RefRenderer::render()\n");
    // render all circles

	  printf("num of circles =  %d \n",numCircles);

    for (int circleIndex=0; circleIndex<numCircles; circleIndex++) {

        int index3 = 3 * circleIndex;

        float px = position[index3];
        float py = position[index3+1];
        float pz = position[index3+2];
        float rad = radius[circleIndex];

        // compute the minimum and maximum cordinates of circle
        float minX = px - rad;
        float maxX = px + rad;
        float minY = py - rad;
        float maxY = py + rad;

        // convert normalized coordinate bounds to integer screen

        int screenMinX = CLAMP(static_cast<int>(minX * image->width), 0, image->width);
        int screenMaxX = CLAMP(static_cast<int>(maxX * image->width)+1, 0, image->width);
        int screenMinY = CLAMP(static_cast<int>(minY * image->height), 0, image->height);
        int screenMaxY = CLAMP(static_cast<int>(maxY * image->height)+1, 0, image->height);

        float invWidth = 1.f / image->width;
        float invHeight = 1.f / image->height;

        // for each pixel in the box, the corresponding pixel on screen is determined.
        for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {

            // pointer to pixel data
            float* imgPtr = &image->data[4 * (pixelY * image->width + screenMinX)];

			//When Coloring the pixel, we treated the pixel as a point at the //center of the pixel. As shading math is configued in Normalized //space, converting the pixels. 
            for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {

                float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
                float pixelCenterNormY = invHeight * (static_cast<float>(pixelY) + 0.5f);
				//Calling the Shader
                pixel_shader(circleIndex, pixelCenterNormX, pixelCenterNormY, px, py, pz, imgPtr);
                imgPtr += 4;
            }
        }
    }
}







