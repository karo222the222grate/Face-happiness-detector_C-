# Face-happiness-detector_C++

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Define constants for easy configuration
const string FACE_CASCADE_PATH = "C:\\Users\\karo\\Documents\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
const string SMILE_CASCADE_PATH = "C:\\Users\\karo\\Documents\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml";
const double SCALE_FACTOR = 1.1;
const int MIN_NEIGHBORS = 4;
const int MIN_FACE_SIZE = 50;

int main() {
    try {
        // Load face and smile cascades
        CascadeClassifier face_cascade, smile_cascade;

        cout << "Checking Haar Cascade files..." << endl;
        if (!fs::exists(FACE_CASCADE_PATH) || !fs::exists(SMILE_CASCADE_PATH)) {
            cerr << "Error: Haar Cascade file not found at specified paths." << endl;
            return -1;
        }

        cout << "Loading Haar Cascades..." << endl;
        if (!face_cascade.load(FACE_CASCADE_PATH) || !smile_cascade.load(SMILE_CASCADE_PATH)) {
            cerr << "Error: Failed to load Haar Cascade classifiers" << endl;
            return -1;
        }
        cout << "Haar Cascades loaded successfully" << endl;

        // Initialize video capture
        cout << "Initializing video capture..." << endl;
        VideoCapture cap(0, CAP_DSHOW); // Try using DirectShow backend
        if (!cap.isOpened()) {
            cap.open(0); // Fallback to default backend if DirectShow fails
            if (!cap.isOpened()) {
                cerr << "Error: Failed to open video capture" << endl;
                return -1;
            }
        }
        cout << "Video capture initialized successfully" << endl;

        Mat frame;
        while (true) {
            if (!cap.read(frame)) {
                cerr << "Warning: Failed to read frame, retrying..." << endl;
                continue;
            }

            if (frame.empty()) {
                cerr << "Warning: Empty frame received" << endl;
                continue;
            }

            // Convert to grayscale
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // Detect faces
            vector<Rect> faces;
            face_cascade.detectMultiScale(gray, faces, SCALE_FACTOR, MIN_NEIGHBORS, 0, Size(MIN_FACE_SIZE, MIN_FACE_SIZE));

            int happy_faces = 0; // Counter for faces with smiles

            // Process each detected face
            for (const auto& face : faces) {
                Mat faceROI = gray(face); // Region of Interest (ROI) for the face

                // Detect smiles within the face ROI
                vector<Rect> smiles;
                smile_cascade.detectMultiScale(faceROI, smiles, 1.7, 22, 0, Size(25, 25));

                // If a smile is detected within the face, consider it a "happy" face
                if (!smiles.empty()) {
                    happy_faces++;
                    rectangle(frame, face, Scalar(0, 255, 0), 2); // Green rectangle for happy face
                }
                else {
                    rectangle(frame, face, Scalar(255, 0, 0), 2); // Blue rectangle for neutral face
                }
            }

            // Calculate happiness percentage
            int total_faces = faces.size();
            double happiness_percentage = total_faces > 0 ? (static_cast<double>(happy_faces) / total_faces) * 100.0 : 0.0;

            // Display happiness percentage on the frame
            string happiness_text = "Happiness: " + to_string(static_cast<int>(happiness_percentage)) + "%";
            putText(frame, happiness_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

            // Show the frame with face and happiness percentage
            imshow("Happiness Detection", frame);

            // Break loop with 'q'
            char c = (char)waitKey(1);
            if (c == 'q' || c == 'Q')
                break;
        }

        cap.release();
        destroyAllWindows();
        return 0;
    }
    catch (const Exception& e) {
        cerr << "OpenCV exception: " << e.what() << endl;
        return -1;
    }
    catch (const exception& e) {
        cerr << "Standard exception: " << e.what() << endl;
        return -1;
    }
}
