package uk.ac.soton.ecs.group;

import ch.akuhn.matrix.DenseVector;
import ch.akuhn.matrix.Vector;
import org.apache.regexp.RE;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.feature.FloatFVComparison;
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
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.comparator.DistanceComparator;
import org.openrdf.query.algebra.Str;

import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

public class Run1 {
    public static void main( String[] args ) {
        try {
            GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<FImage>(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                    ImageUtilities.FIMAGE_READER);

            GroupedDataset<String, VFSListDataset<FImage>, FImage> testing = new VFSGroupDataset<FImage>(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);

            HashMap<Integer, FImage> map = new HashMap<>();
            HashMap<Integer, String> classifications = new HashMap<>();
            int index = 0;
            for (String group : training.getGroups())
            {
                for (FImage image2 : training.get(group)) {
                    // cropping code
                    FImage image = image2.clone();
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
                    } else {
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

//                    image = ResizeProcessor.zoomInplace(image, 16,16);
                    map.put(index, image);
                    classifications.put(index, group);
                    index++;
                }
            }
            for (FImage image : map.values()) {
                if (image.pixels.length != image.pixels[0].length)
                    System.out.println("DONT WORK");
            }

            ResizeProcessor.zoomInplace(map.get(1), 16,16);

            System.out.println(map.size());

            for (Integer num : map.keySet()) {
                map.put(num, ResizeProcessor.resizeMax(map.get(num).clone(), 16));
            }

            for (FImage image : map.values()) {
                System.out.println(image.pixels.length);
                System.out.println(image.pixels[0].length);
            }

            // KNN stuff

            float[][] combined = new float[training.size()][16*16];
            int i = 0;
            for (FImage image : map.values()) {
                combined[i] = image.getFloatPixelVector();
                i++;
            }
            FloatNearestNeighboursExact blah = new FloatNearestNeighboursExact(combined);
//            blah.searchKNN(query, k);

        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public FImage cropImage(FImage image) {
        int size1 = image.pixels.length;
        int size2 = image.pixels[0].length;
        if (size1 > size2) {
            float[][] cropped = new float[size2][size2];
            for (int i = 0; i < size2; i++) {
                for (int j = 0; j < size2; j++) {
                    cropped[i][j] = image.pixels[i][j];
                }
            }
            return new FImage(cropped);
        }
        else {
            float[][] cropped2 = new float[size1][size1];
            for (int i = 0; i < size1; i++) {
                for (int j = 0; j < size1; j++) {
                    cropped2[i][j] = image.pixels[i][j];
                }
            }
            return new FImage(cropped2);
        }
    }
}
