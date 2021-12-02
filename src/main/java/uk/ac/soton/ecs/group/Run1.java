package uk.ac.soton.ecs.group;

import ch.akuhn.matrix.DenseVector;
import ch.akuhn.matrix.Vector;
import org.apache.regexp.RE;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.knn.FloatNearestNeighboursExact;

import java.net.URL;

public class Run1 {
    public static void main( String[] args ) {
        try {
            GroupedDataset<String, VFSListDataset<FImage>, FImage> images = new VFSGroupDataset<FImage>(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                    ImageUtilities.FIMAGE_READER);


            for (FImage image : images)
            {
                // cropping code
                int size1 = image.pixels.length;
                int size2 = image.pixels[0].length;
                if (size1 > size2) {
                    float[][] cropped = new float[size2][size2];
                    for (int i = 0; i < size2; i++) {
                        for (int j = 0; j < size2; j++) {
                            cropped[i][j] = image.pixels[i][j];
                        }
                    }
                    image.pixels = cropped;
                    if (image.pixels.length != image.pixels[0].length)
                        System.out.println("Bruh");
                }
                else {
                    float[][] cropped2 = new float[size1][size1];
                    for (int i = 0; i < size1; i++) {
                        for (int j = 0; j < size1; j++) {
                            cropped2[i][j] = image.pixels[i][j];
                        }
                    }
                    image.pixels = cropped2;
                    if (image.pixels.length != image.pixels[0].length)
                        System.out.println("Bruh");
                }
                // shrinking code
            }
            for (FImage image : images) {
                ResizeProcessor.zoomInplace(image, 16,16);
            }

            // KNN stuff
            for (String s : images.getGroups()) {
                float[][] combined = new float[images.get(s).size()][];
                int i = 0;
                for (FImage image : images.get(s)) {
                    float[] imageData = image.getPixelVectorNative(new float[image.pixels.length * image.pixels.length]);
                    combined[i] = imageData;
                    i++;
                }
                FloatNearestNeighboursExact blah = new FloatNearestNeighboursExact(combined);
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
