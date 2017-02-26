struct Image;

typedef enum {
    CIRCLE_RGB,
    SNOWFLAKES,
	CIRCLE_Rand
}SceneName;

class render_circle{
	
public:
	
	virtual void image_setup() = 0;
	
	virtual void allocImageBuf(int width, int height);
	
	virtual void loadScene(SceneName name) = 0;
	
	virtual void clearImage() = 0;
	
	virtual void render() = 0;
};
