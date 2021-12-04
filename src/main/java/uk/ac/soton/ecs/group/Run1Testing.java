package uk.ac.soton.ecs.group;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import static org.openimaj.feature.FloatFVComparison.EUCLIDEAN;

public class Run1Testing {
    public static void main( String[] args ) throws IOException {
        GroupedDataset<String, VFSListDataset<Record<FImage>>, Record<FImage>> training = new VFSGroupDataset(
                "D:\\Downloads\\training",
                ImageUtilities.FIMAGE_READER);

        //from chapter 12
        GroupedDataset<String, VFSListDataset<Record<FImage>>, Record<FImage>> allData =
                Caltech101.getData(ImageUtilities.FIMAGE_READER);

        GroupedDataset<String, VFSListDataset<FImage>, FImage> testing = new VFSGroupDataset(
                "D:\\Downloads\\testing",
                ImageUtilities.FIMAGE_READER);


        //second display line works but first one doesn't - don't know what is wrong
//        DisplayUtilities.display(training.getRandomInstance().getImage());
//        DisplayUtilities.display(allData.getRandomInstance().getImage());



        training.remove("training"); // Idk why but when doing a for each over the groups there's an extra group with everything

        //extractor and KNNAnnotator
        TinyImagesExtractor extractor = new TinyImagesExtractor();
        KNNAnnotator myKNNAnnotator = new KNNAnnotator(extractor, EUCLIDEAN, 5);
        myKNNAnnotator.train(training);


        //result thingy from ch12 to test
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(myKNNAnnotator,testing,new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }

    static class TinyImagesExtractor implements FeatureExtractor<float[], FImage> {
        @Override
        public float[] extractFeature(FImage object) {
            FImage image = object;

            //make image square
            if (image.width > image.height)
                image = image.extractCenter(image.height, image.height);
            else
                image = image.extractCenter(image.width, image.width);

            //resize image to 16x16
            image = ResizeProcessor.zoomInplace(image, 16,16);

            //return pixel values packed into a vector
            return image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()]);
        }
    }

    //put predictions for each image in a hashmap and run this method with the hashmap as the argument
    public static void outputResult(HashMap<String,String> predictions) throws FileNotFoundException {
        PrintWriter myWriter = new PrintWriter("run1.txt");

        for (String fileName : predictions.keySet()) {
            myWriter.println(fileName + predictions.get(fileName));
        }
    }
}
