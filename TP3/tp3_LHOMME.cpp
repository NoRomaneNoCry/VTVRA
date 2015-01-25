/** Romane LHOMME - 25 janvier 2015 **/
/** Vision tridimentionnelle, vidéo et réalite augmentée - TP3 **/

#include <stdio.h>   

////////////////////////////////////////////////////////////////////////////////////////////
// Fonctions OpenCV 
////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>

#ifdef WIN32
	// unix to win porting
	#define	  __func__   __FUNCTION__
	#define	  __PRETTY_FUNCTION__   __FUNCTION__
	#define   snprintf   _snprintf
	#define	  sleep(n_seconds)   Sleep((n_seconds)*1000)
	#define   round(d)   floor((d) + 0.5)
#endif

#include "Apicamera/src/cameraUVC.h"
#include "Calibration/src/chessboardcalibration.h"
#include "Apicamera/src/cameraOPENCV.h"
#include <iostream>
#include <fstream>

void cameraUVC_getFrame( apicamera::CameraUVC *camera, cv::Mat *out1)
{
	cv::Mat(camera->get1Frame()).copyTo(*out1);
}

void showImage( const char* windowName, const cv::Mat *in)
{
	if( in == NULL || ( in->cols == 0 && in->rows == 0 ) )
	{
		// invalid image, display empty image
		const int w = 200;
		const int h = 100;
		cv::Mat img = cv::Mat( h, w, CV_8UC3, cv::Scalar(0));
		cv::line( img, cv::Point( 0, 0), cv::Point( w-1, h-1), cv::Scalar(0,0,255), 2);
		cv::line( img, cv::Point( 0, h-1), cv::Point( w-1, 0), cv::Scalar(0,0,255), 2);
		cv::imshow( windowName, img);
		return;
	}
	else if( in->depth() == CV_32F  ||  in->depth() == CV_64F )
	{
		// float image must be normalized in [0,1] to be displayed
		cv::Mat img;
		cv::normalize( *in, img, 1.0, 0.0, cv::NORM_MINMAX);
		cv::imshow( windowName, img);
		return;
	}
	cv::imshow( windowName, *in);
}

class ExtrinsicChessboardCalibrator
{
public:
	ExtrinsicChessboardCalibrator( unsigned int _cbWidth, unsigned int _cbHeight, float _squareSize, const char *_intrinsicFileName, const char *_extrinsicFileName)
	{
		// load intrinsic parameters
		camera = new apicamera::CameraOPENCV();
		camera->loadIntrinsicParameters(_intrinsicFileName);

		// initialize calibration
		calibrator = new ChessboardCalibration( camera, 1, _cbWidth, _cbHeight, _squareSize);
		extrinsicFileName = _extrinsicFileName;
	}

	~ExtrinsicChessboardCalibrator()
	{
		delete calibrator;
		delete camera;
	}

	void processFrame( const cv::Mat *inImg, const cv::Mat *intrinsicA, const cv::Mat *intrinsicK, cv::Mat *translation, cv::Mat *rotation, cv::Mat *error, cv::Mat *outImg)
	{
		if( (! inImg) || (! inImg->data) )
			return;

		inImg->copyTo(*outImg);
		IplImage currentImage(*outImg);

		// set intrinsic parameters if provided through block inputs
		if( intrinsicA )
		{
			cv::Mat A( 3, 3, CV_32FC1, camera->intrinsicA);
			intrinsicA->copyTo(A);
		}
		if( intrinsicK )
		{
			cv::Mat K( 1, 4, CV_32FC1, camera->intrinsicK);
			intrinsicK->copyTo(K);
		}

		// compute extrinsic parameters
		camera->extrinsicError = calibrator->findExtrinsicParameters( 0.0f, 0.0f, &currentImage);

		// save extrinsic parameters to file
		camera->saveExtrinsicParameters(extrinsicFileName);

		// copy extrinsic parameters and error to outputs
		cv::Mat T( 1, 3, CV_32FC1, camera->extrinsicT);
		T.copyTo(*translation);
		cv::Mat R( 3, 3, CV_32FC1, camera->extrinsicR);
		R.copyTo(*rotation);
		cv::Mat E( 1, 1, CV_32FC1, &(camera->extrinsicError));
		E.copyTo(*error);
	}

protected:
	// camera is used only to store/load/save intrinsic/extrinsic parameters
	apicamera::CameraOPENCV *camera;

	ChessboardCalibration *calibrator;
	const char *extrinsicFileName;	
};

// Variables globales OpenCV
apicamera::CameraUVC cameraUVC; // La camèra
apicamera::OpenParameters openParam; // Les paramètres de la caméra
ExtrinsicChessboardCalibrator * extrinsicCalibrator; // Le calibrateur pour les paramètres extrinsèques

////////////////////////////////////////////////////////////////////////////////////////////
// Fonctions OpenGL
////////////////////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <locale.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

// Variables globales OpenGL
int Window_ID;
int windowWidth = 640;
int windowHeight = 480;

int mireWidth; // Nombre de carrés en largeur
int mireHeight; // Nombre de carrés en hauteur
int mireSize; // Taille en cm d'un carré
char * intrinsicParams; // Nom du fichier contenant les paramètres intrinsèques

GLuint texture; // Texture permettant de stocker ce que voit la caméra OpenCV

void unproject(const float * A, const float * K, const float * pi, float * pc) 
{
	// On résout le système pi = A * pc
	pc[0] = (pi[0] - A[2]) / A[0];
	pc[1] = (pi[1] - A[5]) / A[4];
	pc[2] = 1.0f;
}

float distance(const float * a, const float * b)
{
	return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2));
}

void calculerFrustum( const float *A, const float *K, float w, float h, float *frustum)
{
	float dirmid[3], dirleft[3], dirtop[3], dirright[3], dirdown[3];
	float mid[2] = {w/2.f, h/2.f}, left[2] = {0.0f, h/2.f}, top[2] = {w/2.f, h}, right[2] = {w, h/2.f}, down[2] = {w/2.f, 0.0f};
	
	unproject(A, K, mid, dirmid); 
	unproject(A, K, left, dirleft); 
	unproject(A, K, top, dirtop); 
	unproject(A, K, right, dirright); 
	unproject(A, K, down, dirdown);
		
	frustum[0] = -1.0f * distance(dirmid, dirleft); // left
	frustum[1] =  distance(dirmid, dirright); // right
	frustum[2] = -1.0f * distance(dirmid, dirdown); // bottom
	frustum[3] =  distance(dirmid, dirtop); // top
	frustum[4] = 1.0f; // near
	frustum[5] = 1000.0f; // far
}

void dessineAxes(float taille)
{
	glBegin(GL_LINES);
	
	glColor3f(1.0f,0.0f,0.0f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(taille,0.0f,0.0f);
	glColor3f(0.0f,1.0f,0.0f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(0.0f,taille,0.0f);
	glColor3f(0.0f,0.0f,1.0f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(0.0f,0.0f,taille);

	glEnd();
}

void dessineMire( int w, int h, float sz)
{
	int i, j;
	glBegin(GL_QUADS);    	
	for(i = -1; i < w-1; i++) {
		for(j = -1; j < h-1; j++) {
			// couleur du carré de la mire
			((i+j)%2 == 0) ? glColor3f(0.0f,0.0f,0.0f) : glColor3f(1.0f,1.0f,1.0f);
			
			glVertex3f(i*sz, j*sz, 0.0f);
			glVertex3f(i*sz, (j+1)*sz, 0.0f);
			glVertex3f((i+1)*sz, (j+1)*sz, 0.0f);
			glVertex3f((i+1)*sz, j*sz, 0.0f);
		}
	}
	glEnd();
}

void calculerTransformation( const float *R, const float *T, float *GtoC)
{
	// On transmet la matrice de rotation dans le bon ordre à OpenGL
	// car OpenGL travaille sur les colonnes
	GtoC[0] = R[0];
	GtoC[1] = R[3];
	GtoC[2] = R[6];
	GtoC[3] = 0.0f;

	GtoC[4] = R[1];
	GtoC[5] = R[4];
	GtoC[6] = R[7];
	GtoC[7] = 0.0f;

	GtoC[8] = R[2];
	GtoC[9] = R[5];
	GtoC[10] = R[8];
	GtoC[11] = 0.0f;

	GtoC[12] = T[0];
	GtoC[13] = T[1];
	GtoC[14] = T[2];
	GtoC[15] = 1.0f;
}

void calculerDirection(const float *A, const float *K, float w, float h, float *direction)
{
	// Même principe que Unproject sauf qu'ici le point que l'on veut déprojeter
	// est le centre de l'image
	float a = w/2.f, b = h/2.f, c = 1.f;
	
	direction[2] = c ;
	direction[1] = (b - A[5]) / A[4];
	direction[0] = (a - A[2]) / A[0];
}

void glDrawFromCamera( const float *A, const float *K, const float *R, const float *T) 
{
	
    glEnable(GL_TEXTURE_2D); 
    glDisable(GL_DEPTH_TEST); 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

    /*** AFFICHAGE DU FOND ***/
    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity(); 
    gluOrtho2D(0,1 ,0 ,1); 
    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity();

    cv::Mat img; 
    cameraUVC_getFrame(&cameraUVC, &img);

    cv::flip(img,img,0);
    // On génère à partir de la frame de la caméra la texture qui sera utilisée comme fond  
    glGenTextures(1, &texture); 
    glBindTexture(GL_TEXTURE_2D, texture); 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // On colle la texture sur un Quad d'OpenGL
    glBegin(GL_QUADS); 
    glTexCoord2d(0,0); 
    glVertex2f(0,0); 
    glTexCoord2d(1,0); 
    glVertex2f(1,0); 
    glTexCoord2d(1,1); 
    glVertex2f(1,1); 
    glTexCoord2d(0,1); 
    glVertex2f(0,1); 
    glEnd();

    glDisable(GL_TEXTURE_2D); 
	/*** FIN DE L'AFFICHAGE DU FOND ***/

    // Pour cacher avec la mire virtuelle et la théière ce qui est vu par la caméra
    glEnable(GL_DEPTH_TEST);

    /*** AFFICHAGE DE LA MIRE VIRTUELLE ***/
	float frustum[6], GtoC[16], direction[3];
	
	calculerFrustum( A, K, windowWidth, windowHeight, frustum);
	calculerDirection(A, K, windowWidth, windowHeight, direction);
	calculerTransformation(R, T, GtoC);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum( frustum[0], frustum[1], frustum[2], frustum[3], frustum[4], frustum[5]);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f, 0.0f, 0.0f, direction[0], direction[1], direction[2], 0.0f, -1.0f, 0.0f);
	glMultMatrixf(GtoC);
	
	dessineAxes(8.0);
	dessineMire(mireWidth, mireHeight, mireSize);
	/*** FIN DE L'AFFICHAGE DE LA MIRE VIRTUELLE ***/

	/*** AFFICHAGE DE LA THEIERE ***/
	// Affichage au niveau de l'origine du repère, en gris et en maillage
	glColor3f(0.5f,0.4f,0.5f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glRotatef(-90, 1.0f, 0.0f, 0.0f);
	glTranslatef(0.0f,3.1,0.0f);
	glutSolidTeapot(3);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	/*** FIN DE L'AFFICHAGE DE LA THEIERE ***/
}

void cbRenderScene(void)
{
	cv::Mat mat1;
	cameraUVC_getFrame( &cameraUVC, &mat1);
	cv::Mat R, T, error, output;
	extrinsicCalibrator->processFrame(&mat1, NULL, NULL, &T, &R, &error, &output);
	// Ce que voit la caméra avec le repérage de la mire
	showImage("Camera", &output);
	
	glDrawFromCamera(cameraUVC.intrinsicA, cameraUVC.intrinsicK, (float*)R.data, (float*)T.data);
	glutSwapBuffers();
	// Pour permettre à OpenCV de fonctionner
	cv::waitKey(25);
}

void cbKeyPressed( unsigned char key, int x, int y)
{
	switch (key) 
	{
		case 113: case 81: case 27: // Q (Escape) - We're outta here.
			glutDestroyWindow(Window_ID);
			exit(1);
			break; // exit doesn't return, but anyway...

		case 'i': 
		case 'I':
			break;

		default:
			printf ("KP: No action for %d.\n", key);
			break;
	}
}
 
void ourInit(void) 
{
	// Color to clear color buffer to.
	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

	// Depth to clear depth buffer to; type of test.
	glClearDepth(1.0);
	glDepthFunc(GL_LESS); 

	// Enables Smooth Color Shading; try GL_FLAT for (lack of) fun.
	glShadeModel(GL_SMOOTH);
}

////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
////////////////////////////////////////////////////////////////////////////////////////////

int main(  int argc,  char **argv)
{
	if(argc < 5) {
		printf("Pas assez de paramètres en ligne de commande. Exiting ...\n");
		printf("Usage : ./tp3_LHOMME nom_fichier_intrinseques largeur_mire hauteur_mire taille_carre_mire\n");
		exit(1);
	}
	// Paramètres en ligne de commande
	mireWidth = atoi(argv[2]);
	mireHeight = atoi(argv[3]);
	mireSize = atoi(argv[4]);
	intrinsicParams = argv[1];
	
	setbuf(stdout, NULL);
	
	// initialize camera UVC
	openParam.width = 640;
	openParam.height = 480;
	openParam.fRate = 30;
	
	if( cameraUVC.open( 0, &openParam) != 0 )
	{
		printf( "failed to init UVC Camera. Exiting ...\n");
		exit(1);
	}
	
	cameraUVC.loadIntrinsicParameters(intrinsicParams);
	extrinsicCalibrator = new ExtrinsicChessboardCalibrator( mireWidth, mireHeight, mireSize, intrinsicParams, "./extrinsics.txt");
	
	// pour eviter pb de . et , dans les floats
	setlocale(LC_NUMERIC, "C");	

	// initialisation de glut
	glutInit(&argc, argv);

	// To see OpenGL drawing, take out the GLUT_DOUBLE request.
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(windowWidth, windowHeight);

	// Open a window 
	Window_ID = glutCreateWindow("OpenGL");
	

	// Register the callback function to do the drawing. 
	glutDisplayFunc(&cbRenderScene);

	// If there's nothing to do, draw.
	glutIdleFunc(&cbRenderScene);

	// And let's get some keyboard input.
	glutKeyboardFunc(&cbKeyPressed);

	// OK, OpenGL's ready to go.  Let's call our own init function.
	ourInit();

	// Pass off control to OpenGL.
	// Above functions are called as appropriate.
	glutMainLoop();

	return 1;
}

