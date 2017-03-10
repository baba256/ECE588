#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
//#include "render_circle.h"
#include "cuda_renderer.h"

//This function is used to compare timings
void Check_Render_timing(CircleRenderer* ref_renderer, CircleRenderer* cuda_renderer,
                        int benchmarkFrameStart, int totalFrames, const std::string& frameFilename);
						
int main(int argc, char** argv)
{
	int benchMarkStart 	= -1;
	int benchMarkEnd  	= -1;
	int imageSize 		= 768;
	
	std::string 	sceneNameStr;
	std::string 	frameFilename;
	SceneName 		sceneName;
	
	if (optind >= argc) {
       // printf(stderr, "Error: missing scene name\n");
		printf( "Error: missing scene name\n");
      //  usage(argv[0]);
        return 1;
    }
	
	sceneNameStr = argv[optind];
	//Take the file from command line
	frameFilename = argv[2]; //FIXME: Check this
	
	if(sceneNameStr.compare("snow") == 0){
		sceneName = SNOWFLAKES;
		
	} else if (sceneNameStr.compare("rgb") == 0) {
		sceneName = CIRCLE_RGB;
		
	}else if (sceneNameStr.compare("randcircle") == 0){
		sceneName = CIRCLE_Rand;
		
	} else {
		fprintf(stderr,"Unknown Scene Name (%s) \n",sceneNameStr.c_str());
		return 1;
	}
	
	printf("Rendering to %d x %d image\n", imageSize, imageSize);
	
	Render_circle* ref_renderer;
	Render_circle* cuda_render;
	
	ref_renderer	= new RefRenderer();
	cuda_render 	= new Cuda_renderer();
	
	//Initializing to render using CPU
   ref_renderer->allocImageBuf(imageSize,imageSize);
   ref_renderer->loadScene(sceneName);
   ref_renderer->setup();
   
	//Initializing to render using GPU
   cuda_render->allocImageBuf(imageSize, imageSize);
   cuda_render->loadScene(sceneName);
   cuda_render->setup();
   
	//Calling the timing check
	Check_Render_timing(ref_renderer,cuda_render,0,1,frameFilename);
   
	return 0;
}
