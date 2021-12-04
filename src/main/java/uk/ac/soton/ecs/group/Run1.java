package uk.ac.soton.ecs.group;

import ch.akuhn.matrix.DenseVector;
import ch.akuhn.matrix.Vector;
import org.apache.regexp.RE;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.FImage2FloatFV;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.comparator.DistanceComparator;
import org.openimaj.util.pair.IntFloatPair;
import org.openrdf.query.algebra.Str;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Run1 {
    //TODO Figure out KNN implementation
    //TODO Figure out how to output prediction file
    public static void main( String[] args ) {
        try {

            // Data set imports
            GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                    ImageUtilities.FIMAGE_READER);

            GroupedDataset<String, VFSListDataset<FImage>, FImage> testing = new VFSGroupDataset<FImage>(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);

            training.remove("training"); // Idk why but when doing a for each over the groups there's an extra group with everything


            // Training

            // Manual implementation
//            HashMap<Integer, String> classificationMap = new HashMap<>();
//            int trainingSize = 0;
//            for (String s : training.getGroups()) {
//                trainingSize += training.get(s).size();
//            }
//            float[][] imageVectors = new float[trainingSize][16 * 16];
//
//            int i = 0;
//            for (String group : training.getGroups()) {
//                System.out.println(training.get(group).size());
//                for (FImage image : training.get(group)) {
//                    imageVectors[i] = extractTinyImage(image).getFloatPixelVector();
//                    classificationMap.put(i, group);
//                    i++;
//                }
//            }
//            FloatNearestNeighboursExact fNN = new FloatNearestNeighboursExact(imageVectors);

            // KNNAnnotator implementation
            KNNAnnotator<FImage, String, FloatFV> classifier = new KNNAnnotator<>(new FImage2FloatFV(), FloatFVComparison.EUCLIDEAN);


            // Prediction


        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String classifyImage(FloatNearestNeighboursExact classifier, HashMap<Integer, String> mapping,FImage image) {
        float[] query = extractTinyImage(image).getFloatPixelVector();
        List<IntFloatPair> nearestNeighbours = classifier.searchKNN(query, 5);

        List<String> neighbourClassifications = new ArrayList<>();
        for (IntFloatPair pair : nearestNeighbours) {
            neighbourClassifications.add(mapping.get(pair.first));
        }

        int mode = 0;
        String modalValue = null;
        for (String classification : neighbourClassifications) {
            int count = 0;
            for (String s : neighbourClassifications) {
                if (s.equals(classification))
                    count++;
            }
            if (count > mode) {
                mode = count;
                modalValue = classification;
            }
        }
        return modalValue;
    }

    private static FImage extractTinyImage(FImage image) {

//        System.out.println(image.width + " x " + image.height);


        FImage out = new FImage(image.width, image.height);
        if (image.width > image.height)
            out = image.extractCenter(image.height, image.height);
        else
            out = image.extractCenter(image.width, image.width);

//        System.out.println(out.width + " x " + out.height);

        return ResizeProcessor.zoomInplace(out, 16,16);
    }
}
