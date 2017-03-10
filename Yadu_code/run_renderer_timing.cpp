//This file is only for seeing the timings
#include <string>
#include <math.h>
#include <stdio.h>
#include "render_circle.h"
#include "cycleTimer.h"
#include "image.h"
#include "ppm.h"



void Check_Render_timing_cuda(Render_circle* ref_renderer, Render_circle* cuda_renderer,
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

		if (frame == startFrame)
			startTime = CycleTimer::currentSeconds();

		double startClearTime = CycleTimer::currentSeconds();
		cuda_renderer->clearImage();
		double endClearTime = CycleTimer::currentSeconds();

		double startRenderTime = CycleTimer::currentSeconds();
		//ref_renderer->render();
		cuda_renderer->render();
		double endRenderTime = CycleTimer::currentSeconds();

		if (frame >= startFrame) {
			double startFileSaveTime = CycleTimer::currentSeconds();
			if (dumpFrames) {
				printf("Dumping frames\n");
				char filename[1024];
				sprintf(filename, "%s_%04d.cuda_ppm", frameFilename.c_str(), frame);
				writePPMImage(cuda_renderer->image_setup(), filename);

			}
			double endFileSaveTime = CycleTimer::currentSeconds();

			totalClearTime += endClearTime - startClearTime;
			//totalAdvanceTime += endAdvanceTime - endClearTime;
			totalRenderTime += endRenderTime ;
			totalFileSaveTime += endFileSaveTime - endRenderTime;

		}
	}
	printf("Overall:  Completed writing to PPM \n");
	double endTime = CycleTimer::currentSeconds();
	totalTime = endTime - startTime;

	printf("Clear:    %.4f ms\n", 1000.f * totalClearTime / totalFrames);
	//printf("Advance:  %.4f ms\n", 1000.f * totalAdvanceTime / totalFrames);
	printf("Render:   %.4f ms\n", 1000.f * totalRenderTime / totalFrames);
	printf("Total:    %.4f ms\n", 1000.f * (totalClearTime + totalRenderTime) / totalFrames);
	if (dumpFrames)
		printf("File IO:  %.4f ms\n", 1000.f * totalFileSaveTime / totalFrames);
	printf("\n");
	printf("Overall:  %.4f sec (note units are seconds)\n", totalTime);
}

void Check_Render_timing_ref(Render_circle* ref_renderer, Render_circle* cuda_renderer,
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

		if (frame == startFrame)
			startTime = CycleTimer::currentSeconds();

		double startClearTime = CycleTimer::currentSeconds();
		ref_renderer->clearImage();
		double endClearTime = CycleTimer::currentSeconds();

		double startRenderTime = CycleTimer::currentSeconds();
		ref_renderer->render();
		double endRenderTime = CycleTimer::currentSeconds();

		if (frame >= startFrame) {
			double startFileSaveTime = CycleTimer::currentSeconds();
			if (dumpFrames) {
				printf("Dumping frames\n");
				char filename[1024];
				sprintf(filename, "%s_%04d.ref_ppm", frameFilename.c_str(), frame);
				writePPMImage(ref_renderer->image_setup(), filename);

			}
			double endFileSaveTime = CycleTimer::currentSeconds();

			totalClearTime += endClearTime - startClearTime;
		//	totalAdvanceTime += endAdvanceTime - endClearTime;
			totalRenderTime += endRenderTime;
			totalFileSaveTime += endFileSaveTime - endRenderTime;

		}
	}
	printf("Overall:  Completed writing to PPM \n");
	double endTime = CycleTimer::currentSeconds();
	totalTime = endTime - startTime;

	printf("Clear:    %.4f ms\n", 1000.f * totalClearTime / totalFrames);
	//printf("Advance:  %.4f ms\n", 1000.f * totalAdvanceTime / totalFrames);
	printf("Render:   %.4f ms\n", 1000.f * totalRenderTime / totalFrames);
	printf("Total:    %.4f ms\n", 1000.f * (totalClearTime + totalRenderTime) / totalFrames);
	if (dumpFrames)
		printf("File IO:  %.4f ms\n", 1000.f * totalFileSaveTime / totalFrames);
	printf("\n");
	printf("Overall:  %.4f sec (note units are seconds)\n", totalTime);
}

