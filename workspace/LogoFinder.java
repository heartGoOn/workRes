package com.example.z675540.tryndk;



import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;


import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.perspectiveTransform;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.imgproc.Imgproc.blur;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class LogoFinder {

    private Mat rawLogo;
    private FeatureDetector detector;
    private DescriptorExtractor extractor;
    private Mat processedLogo;
    private MatOfKeyPoint logoKeyPoints;
    private Mat logoDescriptors;
    private DescriptorMatcher matcher;

    public LogoFinder(Mat logo, int threshold) {
        this.rawLogo = logo;
        this.logoKeyPoints = new MatOfKeyPoint();
        this.logoDescriptors = new Mat();
        this.detector = FeatureDetector.create(FeatureDetector.SURF);
        this.extractor  = DescriptorExtractor.create(DescriptorExtractor.SURF);
        this.matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
    }

    public void init() {
        processedLogo = preProcess(this.rawLogo);
        detector.detect(processedLogo, logoKeyPoints);
        extractor.compute(processedLogo, logoKeyPoints, logoDescriptors);
    }

    public List<Point> findLogo(Mat scene, int dTop, int dBottom, int dLeft, int dRight) {
        //构造mask
        Mat mask = Mat.zeros(new Size(scene.cols(), scene.rows()), CV_8U);

        Mat maskROI = mask.submat(new Rect(dTop, dLeft, scene.cols() - dRight - dLeft, scene.rows() - dBottom - dTop));
        maskROI.setTo(new Scalar(1));

        Mat processedScene = preProcess(scene);
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        Mat sceneDescriptors = new Mat();
        detector.detect(processedScene, sceneKeyPoints, mask);
        extractor.compute(processedScene, sceneKeyPoints, sceneDescriptors);

        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(logoDescriptors, sceneDescriptors, matches);

        double max_dist = 0;
        double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < logoDescriptors.rows(); i++) {
            double dist = matches.toArray()[i].distance;
            if (dist < min_dist) {
                min_dist = dist;
            }
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        System.out.println("-- Max dist : %f \n" + max_dist);
        System.out.println("-- Min dist : %f \n" + min_dist);

        List<DMatch> goodMatches = new ArrayList<DMatch>();
        for (int i = 0; i < logoDescriptors.rows(); i++) {
            if (matches.toArray()[i].distance < 3 * min_dist) {
                goodMatches.add(matches.toArray()[i]);
            }
        }

        //-- Localize the object
        List<Point> logoMatchedPoint = new ArrayList<>();
        List<Point> sceneMatchedPoint = new ArrayList<>();

        for (int i = 0; i < goodMatches.size(); i++) {
            //-- Get the keypoints from the good matches
            logoMatchedPoint.add(logoKeyPoints.toArray()[goodMatches.get(i).queryIdx].pt);
            sceneMatchedPoint.add(sceneKeyPoints.toArray()[goodMatches.get(i).trainIdx].pt);
        }

        MatOfPoint2f logoMatchedPointAsMat = new MatOfPoint2f();
        logoMatchedPointAsMat.fromList(logoMatchedPoint);
        MatOfPoint2f sceneMatchedPointAsMat = new MatOfPoint2f();
        sceneMatchedPointAsMat.fromList(sceneMatchedPoint);

        Mat transform = findHomography(logoMatchedPointAsMat, sceneMatchedPointAsMat, RANSAC, 1000 );

        List<Point>  logoCornersAsList = new ArrayList<>();
        logoCornersAsList.add(new Point(0, 0));
        logoCornersAsList.add(new Point(processedLogo.cols(), 0));
        logoCornersAsList.add(new Point(processedLogo.cols(), processedLogo.rows()));
        logoCornersAsList.add(new Point(0, processedLogo.rows()));

        MatOfPoint2f logoCorners = new MatOfPoint2f();
        logoCorners.fromList(logoCornersAsList);
        MatOfPoint2f sceneCorners = new MatOfPoint2f();

        perspectiveTransform(logoCorners, sceneCorners, transform);

        return sceneCorners.toList();
    }

    public List<Point> findLogo(Mat scene) {
        return findLogo(scene, 0, 0, 0, 0);
    }

    private Mat preProcess(Mat mat) {
        Mat processed = new Mat();
        cvtColor(mat, processed, CV_8U);
        blur(processed, processed, new Size(3, 3));
        return processed;
    }

}
