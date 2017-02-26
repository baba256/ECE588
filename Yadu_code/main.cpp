#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
//#include "render_circle.h"
#include "cuda_renderer.h"

int main(int argc, char** argv)
{
	int benchMarkStart 	= -1;
	int benchMarkEnd  	= -1;
	int imageSize 		= 768;
	
	std::string 	sceneNameStr;
	SceneName 		sceneName;
	
	if (optind >= argc) {
       // printf(stderr, "Error: missing scene name\n");
		printf( "Error: missing scene name\n");
      //  usage(argv[0]);
        return 1;
    }
	
	sceneNameStr = argv[optind];
	
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
	

	Render_circle* cuda_render;
	
	cuda_render = new Cuda_renderer();
	
   cuda_render->allocImageBuf(imageSize, imageSize);
   cuda_render->loadScene(sceneName);

   cuda_render->clearImage();
   cuda_render->setup();
	
	return 0;
}
