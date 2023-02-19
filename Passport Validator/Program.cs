using System;
using System.IO;
using System.Linq;
using DlibDotNet;
using Microsoft.Extensions.Configuration;
using OpenCvSharp;
using Passport_Validator.Models;

namespace Passport_Validator
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var result = new PassportValidationResult();
            try
            {
                IConfigurationRoot config = new ConfigurationBuilder().AddJsonFile("config.json", optional: false, reloadOnChange: true).Build();
                string[] images = Directory.GetFiles(config["imagesFolderPath"]);
                foreach (string image in images)
                {
                    using (var fd = Dlib.GetFrontalFaceDetector())
                    using (var sp = ShapePredictor.Deserialize(Path.GetFullPath(@"shape_predictor_68_face_landmarks.dat")))
                    {
                        var img = Dlib.LoadImage<RgbPixel>(image);
                        //find faces
                        var faces = fd.Operator(img);
                        if (faces.Length > 1)
                            throw new Exception("Multiple Faces Detected");
                        else if (faces.Length < 1)
                            throw new Exception("No Face Detected");
                        else
                        {
                            foreach (var face in faces)
                            {
                                // Get area of face
                                decimal imageFaceArea = (decimal)face.Area;
                                decimal imageArea = (decimal)img.Rect.Area;
                                decimal percentageOfFaceArea = (imageFaceArea / imageArea) * 100;
                                bool validPosture = false;
                                if (percentageOfFaceArea > Convert.ToInt32(config["MinimumFaceAreaByPercentage"]))
                                {
                                    var isBackgroundValid = Util.ValidateBackground(image, face.Left, face.Bottom, face.Right, (int)img.Rect.Width,
                                        config["PassportBackgroundColor"], Convert.ToInt32(config["bgThreshold"]), Convert.ToInt32(config["pixelValidityThreshold"]));

                                    if (isBackgroundValid)
                                    {
                                        // find the landmark points for this face
                                        var shape = sp.Detect(img, face);
                                        // build the 3d face model
                                        var model = Util.GetFaceModel();
                                        // get the landmark point we need
                                        var landmarks = new Mat<Point2d>(1, 6,
                                            (from i in new int[] { 30, 8, 36, 45, 48, 54 }
                                             let pt = shape.GetPart((uint)i)
                                             select new OpenCvSharp.Point2d(pt.X, pt.Y)).ToArray());
                                        // build the camera matrix
                                        var cameraMatrix = Util.GetCameraMatrix((int)img.Rect.Width, (int)img.Rect.Height);
                                        // build the coefficient matrix
                                        var coeffs = new Mat<double>(4, 1);
                                        coeffs.SetTo(0);
                                        // find head rotation and translation
                                        Mat rotation = new Mat<double>();
                                        Mat translation = new Mat<double>();
                                        Cv2.SolvePnP(model, landmarks, cameraMatrix, coeffs, rotation, translation);
                                        // find euler angles
                                        var euler = Util.GetEulerMatrix(rotation);

                                        // calculate head rotation in degrees
                                        var yaw = 180 * euler.At<double>(0, 2) / Math.PI;
                                        var pitch = 180 * euler.At<double>(0, 1) / Math.PI;
                                        var roll = 180 * euler.At<double>(0, 0) / Math.PI;
                                        // looking straight ahead wraps at -180/180, so make the range smooth
                                        pitch = Math.Sign(pitch) * 180 - pitch;

                                        // calibrate yaw, roll and pitch
                                        if (yaw >= -4 && yaw <= 4 && pitch >= -15 && pitch <= 15 && roll >= -5 && roll <= 5)
                                        {
                                            validPosture = true;
                                            if (isBackgroundValid)
                                                Dlib.DrawRectangle(img, face, color: new RgbPixel(0, 255, 0), thickness: 4);
                                            else
                                                Dlib.DrawRectangle(img, face, color: new RgbPixel(0, 0, 0), thickness: 2);
                                        }

                                        // create a new model point in front of the nose, and project it into 2d
                                        var poseModel = new Mat<Point3d>(1, 1, 0);
                                        var poseProjection = new Mat<Point2d>();
                                        Cv2.ProjectPoints(poseModel, rotation, translation, cameraMatrix, coeffs, poseProjection);

                                        // draw a line from the tip of the nose pointing in the direction of head pose
                                        var landmark = landmarks.At<Point2d>(0);
                                        var p = poseProjection.At<Point2d>(0);
                                        Dlib.DrawLine(
                                            img,
                                            new DlibDotNet.Point((int)landmark.X, (int)landmark.Y),
                                            new DlibDotNet.Point((int)p.X, (int)p.Y),
                                            color: new RgbPixel(0, 255, 255));

                                        // draw the key landmark points in yellow on the image
                                        foreach (var i in new int[] { 30, 8, 36, 45, 48, 54 })
                                        {
                                            var point = shape.GetPart((uint)i);
                                            var rect = new Rectangle(point);
                                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                                        }

                                        if (isBackgroundValid && validPosture)
                                        {
                                            result.IsValid = true;
                                            result.ErrorMessage = null;
                                        }
                                    }
                                    else
                                        throw new Exception("White background not detected");
                                }
                                else
                                    throw new Exception("Detected face area is below requirement. Square aspect ratio is recommended for best results.");
                            }
                        }
                    }
                }
                Console.WriteLine();
            }
            catch (Exception e)
            {
                result.IsValid = false;
                result.ErrorMessage = e.Message;
                Console.WriteLine(result.ToString());
            }
            Console.ReadLine();
        }
    }
}
