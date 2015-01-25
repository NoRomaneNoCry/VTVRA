
CC := g++

EXEC := tp3_LHOMME

# Environnement OASIS
INC := -I"/Shared/TP_VTDVR/LIRIS-VISION" -I"/usr/include/opencv" -I"/Shared/TP_VTDVR/LIRIS-VISION/ModulesAndBricks"
LDFLAGS := -L"/Shared/TP_VTDVR/LIRIS-VISION/ModulesAndBricks/Apicamera/build" -L"/Shared/TP_VTDVR/LIRIS-VISION/ModulesAndBricks/Calibration/build" -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lapicamera -lcameraUVC -lcameraOPENCV -lcalibration -lGL -lGLU -lglut -Xlinker -rpath="/Shared/TP_VTDVR/LIRIS-VISION/ModulesAndBricks/Apicamera/build:/Shared/TP_VTDVR/LIRIS-VISION/ModulesAndBricks/Calibration/build:"

# Environnement personnel
#INC := -I"/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision" -I"/usr/include/opencv" -I"/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision/ModulesAndBricks"
#LDFLAGS := -L"/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision/ModulesAndBricks/Apicamera/build" -L"/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision/ModulesAndBricks/Calibration/build" -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lapicamera -lcameraUVC -lcameraOPENCV -lcalibration -lGL -lGLU -lglut -Xlinker -rpath="/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision/ModulesAndBricks/Apicamera/build:/home/romane/Master2/VisionTridiVideosRealiteAugmentee/Export-LirisVision/ModulesAndBricks/Calibration/build:"

ifeq ($(DEBUG),yes)
	CFLAGS := -g -pg -Wall -DDEBUG
else
	CFLAGS := -O2 -Wall
endif


all: $(EXEC)

%: %.cpp
	$(CC) -o $@ $(CFLAGS) $(INC) "$@.cpp" $(LDFLAGS)

.PHONY: clean mrproper tags

clean:
	rm $(EXEC)

help:
	@echo
	@echo "make  -> build target in release mode"
	@echo "make DEBUG=yes  -> build target in debug mode"
	@echo "make test  -> to debug Makefile"
	@echo

test:
	@echo CC=$(CC) 
	@echo EXEC=$(EXEC)
	@echo INC=$(INC) 
	@echo LDFLAGS=$(LDFLAGS) 

