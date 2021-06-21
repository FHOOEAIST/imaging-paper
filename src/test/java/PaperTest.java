/*
 * Copyright (c) 2021 the original author or authors.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

import aist.science.aistcv.AistCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openimaj.image.FImage;
import org.testng.annotations.Test;
import science.aist.imaging.api.GenericImageFunction;
import science.aist.imaging.api.ImageFunction;
import science.aist.imaging.api.domain.twodimensional.JavaPoint2D;
import science.aist.imaging.api.domain.wrapper.ChannelType;
import science.aist.imaging.api.domain.wrapper.implementation.ImageFactoryFactory;
import science.aist.imaging.service.core.imageprocessing.draw.circle.DrawCircle;
import science.aist.imaging.service.core.storage.Image2ByteSaver;

import java.util.Random;

/**
 * @author Christoph Praschl
 * @author Andreas Pointner
 */
public class PaperTest {
    @Test
    void testExample1() {
        try (var image = ImageFactoryFactory
                .getImageFactory(short[][][].class)
                .getImage(100, 100, ChannelType.GREYSCALE)) {

            /* (2) Draw a circle on the image */
            var draw = new DrawCircle<short[][][]>();
            draw.setRadius(3);
            draw.setColor(new double[]{255});
            draw.accept(image, new JavaPoint2D(50, 50));
            new Image2ByteSaver().accept(image, "/results/example1.png");
        }
    }

    @Test
    void testExample2() {
        AistCVLoader.loadShared();
        /* (1) Create new OpenIMAJ image */
        Random rand = new Random(768457);
        @lombok.Cleanup
        var input = ImageFactoryFactory
                .getImageFactory(FImage.class)
                .getRandomImage(100, 100, ChannelType.GREYSCALE, rand, 0, 255, false);

        /* (2) Combine multiple OpenCV function calls in one lambda function using a Canny edge detector and an image dilation */
        ImageFunction<Mat, Mat> f = i -> {
            var image = i.getImage();
            var res = new Mat();
            // apply canny edge detector
            Imgproc.Canny(image, res, 15, 125);
            // prepare dilation
            var element = Imgproc.getStructuringElement(
                    Imgproc.CV_SHAPE_RECT,
                    new Size(3, 3),
                    new Point(1, 1)
            );

            // apply dilation
            Imgproc.dilate(res, res, element);

            return ImageFactoryFactory
                    .getImageFactory(Mat.class)
                    .getImage(i.getHeight(), i.getWidth(),
                            ChannelType.BINARY, res);
        };

        /* (3) Wrap the function in a GenericImageFunction */
        var function = new GenericImageFunction<FImage, double[][][], Mat, Mat>(f, Mat.class, double[][][].class);

        /* (4) Apply the function on a non-OpenCV image and create an image represented as double-array */
        @lombok.Cleanup
        var thresholdResult = function.apply(input);
        new Image2ByteSaver().accept(thresholdResult.createCopy(ImageFactoryFactory.getImageFactory(short[][][].class)), "/results/example2.png");
    }
}
