using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace Passport_Validator
{
    internal class Util
    {
        public static bool ValidateBackground(string _imagePath, int faceLeftSide, int yStart, int faceRightSide, int imageWidth, 
            string color, int bgThreshold, int pixelValidityThreshold)
        {
            ColorConverter colorConverter = new ColorConverter();
            Color bgColor = (Color)colorConverter.ConvertFromString(color); 
            int validPixelCount = 0;
            int invalidPixelCount = 0;            
            using (Bitmap myBitmap = new Bitmap(_imagePath))
            using (Image img = Image.FromFile(_imagePath))
            {
                // yStart represents the point in the y-axis where the pixel check will start from
                // faceLeftSide represents the starting point (along x-axis) for pixel check on the left side of the passport, vice versa for faceRightSide
                // Combination of the yStart with appropriate X-axis will give the coordinate of the point/pixel (in the 2D image) where we want to check.
                /* pixel evaluation is done across one-dimension from both the left and right sides of the image.
                   Consequently, on completion of the evaluation of one line, the check will continue on the next line directly above till we reach the top of the image which 
                    would mean that the value of yStart is 1.
                */
                while (yStart > 1)
                {
                    // the GetCenterPoint method will give us the point on the X-axis that is between the right/left sides of the face on the image and the extreme ends of the image (left or right)
                    // This would consequently avoid errors in situations where the rectangle of the detected face is smaller than the actual face. This is just a precautionary routine to avoid reading wrong point
                    // Situations mitigated by this approach include a chubby faces and hairy images.
                    var leftXStart = GetCenterPoint(0, faceLeftSide);
                    var rightXStart = GetCenterPoint(img.Width, faceRightSide);

                    // here, pixel evaluation is done on a line on left and right sides of the face.
                    // leftXStart and rightXStart represent the points on the x axis where the evaluation would start on the left and right sides respectively.
                    /* For example, take that leftXStart is 10, imageWidth is 100 and yStart is 20. We will also assume that the image has height of 100. The pixel evaluation 
                       will start at a cordinate of (10, 20) and continue at a cordinate of (9,20) next. This will continue till the value of leftXStart is 1 which will signify that the
                       evaluation has reached the end of the image on the left side. Same thing applies to the rightXStart but this time evaluation move from left to right.
                    */
                    while (leftXStart > 0 || rightXStart < imageWidth)
                    {
                        if (leftXStart > 0)
                        {
                            var pixelColorDistance = ColourDistance(myBitmap.GetPixel(leftXStart, yStart), bgColor);
                            if (pixelColorDistance >= 0 && pixelColorDistance <= bgThreshold)
                                validPixelCount++;
                            else
                                invalidPixelCount++;
                            leftXStart--;
                        }
                        if (rightXStart < imageWidth)
                        {
                            var pixelColorDistance = ColourDistance(myBitmap.GetPixel(rightXStart, yStart), Color.White);
                            if (pixelColorDistance >= 0 && pixelColorDistance <= bgThreshold)
                                validPixelCount++;
                            else
                                invalidPixelCount++;
                            rightXStart++;
                        }
                    }
                    yStart--;
                }
            }
            var pixelValidity = GetValidPixelByPercentage(validPixelCount, invalidPixelCount);

            if (pixelValidity >= pixelValidityThreshold)
                return true;
            return false;
        }

        private static int GetCenterPoint(int a, int b)
        {
            int c = a <= b ? b - a : a - b;
            var d = Convert.ToInt32(c * 0.5);
            if (a <= b)
                return a + d;
            return a - d;
        }

        private static double ColourDistance(Color e1, Color e2)
        {
            long rmean = ((long)e1.R + (long)e2.R) / 2;
            long r = (long)e1.R - (long)e2.R;
            long g = (long)e1.G - (long)e2.G;
            long b = (long)e1.B - (long)e2.B;
            return Math.Sqrt((((512 + rmean) * r * r) >> 8) + 4 * g * g + (((767 - rmean) * b * b) >> 8));
        }

        private static int GetValidPixelByPercentage(int validPixelCount, int invalidPixelCount)
        {
            var totalPixelCount = validPixelCount + invalidPixelCount;
            return (validPixelCount * 100) / totalPixelCount;
        }

        public static Mat<Point3d> GetFaceModel()
        {
            try
            {
                return new Mat<Point3d>(1, 6,
                new Point3d[] {
                    new Point3d(0.0f, 0.0f, 0.0f),
                    new Point3d(0.0f, -330.0f, -65.0f),
                    new Point3d(-225.0f, 170.0f, -135.0f),
                    new Point3d(225.0f, 170.0f, -135.0f),
                    new Point3d(-150.0f, -150.0f, -125.0f),
                    new Point3d(150.0f, -150.0f, -125.0f)
                });
            }
            catch (Exception e)
            {
                throw e;
            }
        }

        public static Mat<double> GetCameraMatrix(int width, int height)
        {
            try
            {
                return new Mat<double>(3, 3,
                new double[] {
                    width, 0,     width / 2,
                    0,     width, height / 2,
                    0,     0,     1
                });
            }
            catch (Exception e)
            {
                throw e;
            }
        }

        public static Mat<double> GetEulerMatrix(Mat rotation)
        {
            try
            {
                // convert the 1x3 rotation vector to a full 3x3 matrix
                var r = new Mat<double>(3, 3);
                Cv2.Rodrigues(rotation, r);

                // set up some shortcuts to rotation matrix
                double m00 = r.At<double>(0, 0);
                double m01 = r.At<double>(0, 1);
                double m02 = r.At<double>(0, 2);
                double m10 = r.At<double>(1, 0);
                double m11 = r.At<double>(1, 1);
                double m12 = r.At<double>(1, 2);
                double m20 = r.At<double>(2, 0);
                double m21 = r.At<double>(2, 1);
                double m22 = r.At<double>(2, 2);

                // set up output variables
                Euler euler_out = new Euler();
                Euler euler_out2 = new Euler();

                if (Math.Abs(m20) >= 1)
                {
                    euler_out.yaw = 0;
                    euler_out2.yaw = 0;

                    // From difference of angles formula
                    if (m20 < 0)  //gimbal locked down
                    {
                        double delta = Math.Atan2(m01, m02);
                        euler_out.pitch = Math.PI / 2f;
                        euler_out2.pitch = Math.PI / 2f;
                        euler_out.roll = delta;
                        euler_out2.roll = delta;
                    }
                    else // gimbal locked up
                    {
                        double delta = Math.Atan2(-m01, -m02);
                        euler_out.pitch = -Math.PI / 2f;
                        euler_out2.pitch = -Math.PI / 2f;
                        euler_out.roll = delta;
                        euler_out2.roll = delta;
                    }
                }
                else
                {
                    euler_out.pitch = -Math.Asin(m20);
                    euler_out2.pitch = Math.PI - euler_out.pitch;

                    euler_out.roll = Math.Atan2(m21 / Math.Cos(euler_out.pitch), m22 / Math.Cos(euler_out.pitch));
                    euler_out2.roll = Math.Atan2(m21 / Math.Cos(euler_out2.pitch), m22 / Math.Cos(euler_out2.pitch));

                    euler_out.yaw = Math.Atan2(m10 / Math.Cos(euler_out.pitch), m00 / Math.Cos(euler_out.pitch));
                    euler_out2.yaw = Math.Atan2(m10 / Math.Cos(euler_out2.pitch), m00 / Math.Cos(euler_out2.pitch));
                }

                // return result
                return new Mat<double>(1, 3, new double[] { euler_out.yaw, euler_out.roll, euler_out.pitch });
            }
            catch (Exception e)
            {
                throw e;
            }
        }

        private struct Euler
        {
            public double yaw;
            public double pitch;
            public double roll;
        };
    }    
}
