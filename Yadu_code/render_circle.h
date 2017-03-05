#ifndef __RENDER_CIRCLE_H__
#define __RENDER_CIRCLE_H__

struct Image;

typedef enum {
    CIRCLE_RGB,
    SNOWFLAKES,
	CIRCLE_Rand
}SceneName;

class Render_circle {
	
public:
	
	virtual const Image* image_setup()=0;
	
	virtual void allocImageBuf(int width, int height)=0;
	
	virtual void loadScene(SceneName name) = 0;
	
	virtual void clearImage() = 0;
	
	virtual void render() = 0;

	virtual void setup() = 0;
};

#endif
