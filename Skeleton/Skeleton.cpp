//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"


using namespace cv;
using namespace std;


const float calibrationSquare = 0.01905f;
const float arucoSquareDimension = 0.05;
const Size chessboardDimension = Size(6, 9);

const float zNear = 0.05;
const float zFar = 500.0;
Mat intrinsic_Matrix(3, 3, CV_64F);
Mat distortion_coeffs(8, 1, CV_64F);
Mat Projection(4, 4, CV_64FC1);

//ezeket utólag szedtem ki a függvényekbõl

float distanceP2P(Point a, Point b) {
	float d = sqrt(fabs(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)));
	return d;
}
float getAngle(Point s, Point f, Point e) {
	float l1 = distanceP2P(f, s);
	float l2 = distanceP2P(f, e);
	float dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
	float angle = acos(dot / (l1 * l2));
	angle = angle * 180 / 3.147;
	return angle;
}

String intToString(int number) {
	stringstream ss;
	ss << number;
	string str = ss.str();
	return str;
}

bool pairCompare(const pair<float, Point>& i, const pair<float, Point>& j) {
	return i.first < j.first;

}

GLfloat* convertMatrixType(const cv::Mat& m)
{
	typedef double precision;

	Size s = m.size();
	GLfloat* mGL = new GLfloat[s.width * s.height];

	for (int ix = 0; ix < s.width; ix++)
	{
		for (int iy = 0; iy < s.height; iy++)
		{
			mGL[ix * s.height + iy] = m.at<precision>(iy, ix);
		}
	}

	return mGL;
}

void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
{
	typedef double precision;

	projection.at<precision>(0, 0) = 2 * calibration.at<precision>(0, 0) / windowWidth;
	projection.at<precision>(1, 0) = 0;
	projection.at<precision>(2, 0) = 0;
	projection.at<precision>(3, 0) = 0;

	projection.at<precision>(0, 1) = 0;
	projection.at<precision>(1, 1) = 2 * calibration.at<precision>(1, 1) / windowHeight;
	projection.at<precision>(2, 1) = 0;
	projection.at<precision>(3, 1) = 0;

	projection.at<precision>(0, 2) = 1 - 2 * calibration.at<precision>(0, 2) / windowWidth;
	projection.at<precision>(1, 2) = -1 + (2 * calibration.at<precision>(1, 2) + 2) / windowHeight;
	projection.at<precision>(2, 2) = (zNear + zFar) / (zNear - zFar);
	projection.at<precision>(3, 2) = -1;

	projection.at<precision>(0, 3) = 0;
	projection.at<precision>(1, 3) = 0;
	projection.at<precision>(2, 3) = 2 * zNear * zFar / (zNear - zFar);
	projection.at<precision>(3, 3) = 0;


	modelview.at<precision>(0, 0) = rotation.at<precision>(0, 0);
	modelview.at<precision>(1, 0) = rotation.at<precision>(1, 0);
	modelview.at<precision>(2, 0) = rotation.at<precision>(2, 0);
	modelview.at<precision>(3, 0) = 0;

	modelview.at<precision>(0, 1) = rotation.at<precision>(0, 1);
	modelview.at<precision>(1, 1) = rotation.at<precision>(1, 1);
	modelview.at<precision>(2, 1) = rotation.at<precision>(2, 1);
	modelview.at<precision>(3, 1) = 0;

	modelview.at<precision>(0, 2) = rotation.at<precision>(0, 2);
	modelview.at<precision>(1, 2) = rotation.at<precision>(1, 2);
	modelview.at<precision>(2, 2) = rotation.at<precision>(2, 2);
	modelview.at<precision>(3, 2) = 0;

	modelview.at<precision>(0, 3) = translation.at<precision>(0, 0);
	modelview.at<precision>(1, 3) = translation.at<precision>(1, 0);
	modelview.at<precision>(2, 3) = translation.at<precision>(2, 0);
	modelview.at<precision>(3, 3) = 1;

	// This matrix corresponds to the change of coordinate systems.
	static double changeCoordArray[4][4] = { {1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1} };
	static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

	modelview = changeCoord * modelview;
}

void createArucoMarkers() {

	Mat outputMarker;

	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	for (int i = 0; i < 50; i++) {

		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);

		ostringstream convert;
		string imageName = "4x4Marker_";

		convert << imageName << i << ".jpg";

		imwrite(convert.str(), outputMarker);
	}
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLenght, vector<Point3f>& corners) {

	for (int i = 0; i < boardSize.height; i++) {

		for (int j = 0; j < boardSize.width; j++) {

			corners.push_back(Point3f(j * squareEdgeLenght, i * squareEdgeLenght, 0.0f));
		}
	}
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false) {

	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) {

		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found) {
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults) {
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("Looking for corners", *iter);
			waitKey(0);
		}
	}
}

void cameraClibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLenght, Mat& cameraMatrix, Mat& distanceCoefficients) {

	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLenght, worldSpaceCornerPoints[0]);

	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVector, tVector;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVector, tVector);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {

	std::ofstream outStream(name);
	if (outStream) {

		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++) {

			for (int c = 0; c < columns; c++) {
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++) {

			for (int c = 0; c < columns; c++) {
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();

		return true;
	}

	return false;

}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {

	ifstream inStream(name);

	if (inStream) {
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {

				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		//Distance Coefficients

		inStream >> rows;
		inStream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {

				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}

		inStream.close();
		return true;
	}

	return false;
}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients) {
	Mat frame;
	Mat drawToFrame;

	vector<Mat> savedImages;

	//vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(0);

	if (!vid.isOpened()) {
		return;
	}

	int framesPerSecond = 20;

	namedWindow("Webcam", WINDOW_AUTOSIZE);

	while (true) {
		if (!vid.read(frame)) {
			break;
		}

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboardDimension, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found) {
			frame.copyTo(drawToFrame);

			drawChessboardCorners(drawToFrame, chessboardDimension, foundPoints, found);

			imshow("Webcam", drawToFrame);
		}
		else {
			imshow("Webcam", frame);
		}

		char character = waitKey(1000 / framesPerSecond);

		switch (character)
		{
		case ' ':
			//saving images

			if (found) {

				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:
			//start calibration

			if (savedImages.size() > 15) {

				cameraClibration(savedImages, chessboardDimension, calibrationSquare, cameraMatrix, distanceCoefficients);
				saveCameraCalibration("IloveCameraCalibration", cameraMatrix, distanceCoefficients);
				break;
			}
		case 27:
			//exit
			return;
			break;
		}

		//vid.release();
	}
}


int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension) {

	Mat frame;

	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners;

	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	VideoCapture vid(1);

	//ha nem nyitotta meg, akkor visszatér
	if (!vid.isOpened()) {
		return -1;
	}

	namedWindow("Webcam", 1);

	vector<Vec3d> rotationVectors, translationVectors;

	while (true) {
		if (!vid.read(frame)) {
			break;
		}

		//így kell az opengl frame-et megnyitni, hogy átkonvertáljk a dolgokat
		/*cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
		frame.convertTo(frame, CV_32FC3, 1 / 255.0f);*/

		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);


		for (int i = 0; i < markerIds.size(); i++) {
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//namedWindow("live", 1);

		Mat modelview; //img1;

		Mat rvec(3, 1, DataType<double>::type);
		Mat tvec(3, 1, DataType<double>::type);

		modelview.create(4, 4, CV_64FC1);

		
	}

	return 1;
}






struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);
		for (int i = 0; i < 2; i++)
			objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;


void renderBackgroundGL(const cv::Mat& image)
{

	GLint polygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, polygonMode);
	glPolygonMode(GL_FRONT, GL_FILL);
	glPolygonMode(GL_BACK, GL_FILL);


	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	static bool textureGenerated = false;
	static GLuint textureId;
	if (!textureGenerated)
	{
		glGenTextures(1, &textureId);

		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		textureGenerated = true;
	}

	// Copy the image to the texture.
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.size().width, image.size().height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image.data);

	// Draw the image.
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_TRIANGLES);
	glNormal3f(0.0, 0.0, 1.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(0.0, 0.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 0.0, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 0.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, 1.0, 0.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);

	// Clear the depth buffer so the texture forms the background.
	glClear(GL_DEPTH_BUFFER_BIT);

	// Restore the polygon mode state.
	glPolygonMode(GL_FRONT, polygonMode[0]);
	glPolygonMode(GL_BACK, polygonMode[1]);
}


// Initialization, create an OpenGL context
void onInitialization() {
	//glViewport(0, 0, windowWidth, windowHeight);
	//scene.build();

	//std::vector<vec4> image(windowWidth * windowHeight);
	//long timeStart = glutGet(GLUT_ELAPSED_TIME);
	//scene.render(image);
	//
	//long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	//printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	//// copy image to GPU as a texture
	//fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	//// create program for the GPU
	//gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");


}

bool calibrated = false;

// Window has become invalid: Redraw
void onDisplay() {

	cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

	cv::Mat distanceCoefficients;

	if (!calibrated) {
		loadCameraCalibration("IloveCameraCalibration", cameraMatrix, distanceCoefficients);
		calibrated = true;
	}
	

	Mat frame;

	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners;

	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	VideoCapture vid(1);

	//ha nem nyitotta meg, akkor visszatér
	if (!vid.isOpened()) {
		return;
	}

	namedWindow("Webcam", 1);

	vector<Vec3d> rotationVectors, translationVectors;

	if (!vid.read(frame)) {
		return;
	}

	//így kell az opengl frame-et megnyitni, hogy átkonvertáljk a dolgokat
	/*cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
	frame.convertTo(frame, CV_32FC3, 1 / 255.0f);*/


	aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
	aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);


	for (int i = 0; i < markerIds.size(); i++) {
		aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
		aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//namedWindow("live", 1);

	Mat modelview; //img1;

	Mat rvec(3, 1, DataType<double>::type);
	Mat tvec(3, 1, DataType<double>::type);

	modelview.create(4, 4, CV_64FC1);


	/*float curr = getAngle(point1, point2, point3);
	curr = curr / 10;
	curr = 10 - curr;*/

	renderBackgroundGL(frame);
	solvePnP(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);

	cv::Mat rotation;
	cv::Rodrigues(rvec, rotation);


	double offsetA[3][1] = { 9,6,6 };
	Mat offset(3, 1, CV_64FC1, offsetA);
	tvec = tvec + rotation * offset;

	generateProjectionModelview(intrinsic_Matrix, rotation, tvec, Projection, modelview);
	glMatrixMode(GL_PROJECTION);
	GLfloat* projection = convertMatrixType(Projection);
	glLoadMatrixf(projection);
	delete[] projection;

	glMatrixMode(GL_MODELVIEW);
	GLfloat* modelView = convertMatrixType(modelview);
	glLoadMatrixf(modelView);
	delete[] modelView;


	glPushMatrix();
	glColor3f(1.0, 0.0, 0.0);

	//ide kell rajzolni a dolgokat
	glutWireTeapot(10.0);
	glPopMatrix();
	glColor3f(1.0, 1.0, 1.0);

	imshow("Webcam", frame);
	//imshow("live", img1);

	glFlush();
	glutSwapBuffers();

	waitKey(27);
	glutPostRedisplay();



	//ez nem kell már  waitkey(27) miatt
	/*if (waitKey(30) >= 0) {
		break;
	}*/
	

	
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}