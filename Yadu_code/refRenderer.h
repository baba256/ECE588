#include "render_circle.h"

class RefRenderer : public Render_circle {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

public:

    RefRenderer();
    virtual ~RefRenderer();
    const Image* image_setup();

    void setup();

    void loadScene(SceneName name);

    void allocImageBuf(int width, int height);

    void clearImage();

    void render();

    //void dumpParticles(const char* filename);

	void pixel_shader( int Index,
					   float pixelx, float pixely,
					   float p_x,float p_y, float p_z,
					   float* pixelData);
};
