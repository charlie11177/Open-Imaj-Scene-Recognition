package uk.ac.soton.ecs.group;

import ch.akuhn.matrix.DenseVector;
import ch.akuhn.matrix.Vector;
import com.google.common.collect.Lists;
import org.apache.regexp.RE;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
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
import org.openimaj.image.feature.FloatFV2FImage;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.comparator.DistanceComparator;
import org.openimaj.util.pair.IntFloatPair;
import org.openrdf.query.algebra.Str;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.net.URL;
import java.util.*;

public class Run1Testing {
    /*
      TODO
        *Train on whole training set
        *Test on whole testing set
        *Try different k values for best accuracy
     */
    public static void main( String[] args ) {
        try {
            // Data set imports
            GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset(
                    "D:\\Downloads/training",
                    ImageUtilities.FIMAGE_READER);

            VFSListDataset<FImage> testing = new VFSListDataset<FImage>(
                    "D:\\Downloads/testing",
                    ImageUtilities.FIMAGE_READER);

            training.remove("training"); // Idk why but when doing a for each over the groups there's an extra group with everything

            // Splits training set for testing
            GroupedRandomSplitter<String, FImage> trainingSplit = new GroupedRandomSplitter<>(training, 50,0,50);

            // Training
            KNNAnnotator<FImage, String, FloatFV> classifier = new KNNAnnotator<>(new TinyImageExtractor(), FloatFVComparison.EUCLIDEAN);
            classifier.train(trainingSplit.getTrainingDataset());
            classifier.setK(9);

            outputResult(makeClassificationsMap(testing,classifier));

            // Testing
            //result thingy from ch12 to test
            ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                    new ClassificationEvaluator<CMResult<String>, String, FImage>(classifier,trainingSplit.getTestDataset(),new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
            Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);
            System.out.println(result);
            System.out.println(Lists.newArrayList(classifier.classify(trainingSplit.getTestDataset().getRandomInstance().getImage()).getPredictedClasses()).get(0));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Returns image vector after:
    // - Cropping image to square
    // - Resizing to 16x16
    // - Normalising
    // - Zero mean (subtract mean from all values)
    private static class TinyImageExtractor extends FImage2FloatFV {
        @Override
        public FloatFV extractFeature(FImage image) {
            FImage out = new FImage(image.width, image.height);
            if (image.width > image.height)
                out = image.extractCenter(image.height, image.height);
            else
                out = image.extractCenter(image.width, image.width);

            out = ResizeProcessor.zoomInplace(out, 16, 16);
            out.normalise();
            out.processInplace(new MeanCenter());
            return super.extractFeature(out);
        }
    }

    //outputs
    public static void outputResult(HashMap<String,String> predictions) throws FileNotFoundException {
        PrintWriter myWriter = new PrintWriter("run1.txt");

        for (String fileName : predictions.keySet()) {
            myWriter.println(fileName + " " + predictions.get(fileName));
        }
    }

    //calculates classifications for each testing image and puts them in a Hashmap along with the file name
    public static HashMap<String,String> makeClassificationsMap(VFSListDataset<FImage> testingImages,KNNAnnotator classifier) {
        HashMap classifications = new HashMap<String,String>();
        int index = 0;
        for (FImage testImage : testingImages) {
            classifications.put(testingImages.getID(index),Lists.newArrayList(classifier.classify(testImage).getPredictedClasses()).get(0));
            index++;
        }
        return classifications;
    }
}


