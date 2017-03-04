//This file is only for seeing the timings
#include <string>
#include <math.h>

#include "circleRenderer.h"
//#include "cycleTimer.h"
#include "image.h"
#include "ppm.h"


void CheckBenchmark(CircleRenderer* ref_renderer, CircleRenderer* cuda_renderer,
					int startFrame, int totalFrames, const std::string& frameFilename)
{

    double totalClearTime = 0.f;
    double totalRenderTime = 0.f;
    double totalFileSaveTime = 0.f;
    double totalTime = 0.f;
    double startTime= 0.f;

    bool dumpFrames = frameFilename.length() > 0;

    printf("\nRunning benchmark, %d frames, beginning at frame %d ...\n", totalFrames, startFrame);
//    if (dumpFrames)
//        printf("Dumping frames to %s_xxx.ppm\n", frameFilename.c_str());

    for (int frame=0; frame<startFrame + totalFrames; frame++) {

        ref_renderer->clearImage();
      
        cuda_renderer->clearImage();

        ref_renderer->render();
        cuda_renderer->render();

        if (frame >= startFrame) {
            if (dumpFrames) {
                char filename[1024];
                sprintf(filename, "%s_%04d.ppm", frameFilename.c_str(), frame);
                writePPMImage(ref_renderer->image_setup(), filename);
                //renderer->dumpParticles("snow.par");
            }

            double endFileSaveTime = CycleTimer::currentSeconds();
        }
    }
    printf("Overall:  Completed writing to PPM \n");

}