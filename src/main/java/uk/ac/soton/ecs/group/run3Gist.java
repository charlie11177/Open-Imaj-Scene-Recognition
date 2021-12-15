package uk.ac.soton.ecs.group;

import com.google.common.collect.Lists;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.FImage2FloatFV;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;

import java.util.Map;

public class run3Gist {
    public static void main( String[] args ) {
        try {
            // Data set imports
            GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                    ImageUtilities.FIMAGE_READER);

            VFSListDataset<FImage> testing = new VFSListDataset<FImage>(
                    "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);

            training.remove("training"); // Idk why but when doing a for each over the groups there's an extra group with everything

            // Splits training set for testing
            GroupedRandomSplitter<String, FImage> trainingSplit = new GroupedRandomSplitter<>(training, 50,0,50);

            // Training
            Gist gist = new Gist();
            NaiveBayesAnnotator<FImage, String> classifier = new NaiveBayesAnnotator<>(new GistExtractor(gist), NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
            classifier.train(trainingSplit.getTrainingDataset());

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

    private static class GistExtractor extends FImage2FloatFV {
        Gist gist;
        public GistExtractor(Gist gist) {
            this.gist = gist;
        }

        @Override
        public FloatFV extractFeature(FImage image) {
            gist.analyseImage(image);
            return gist.getResponse();
        }
    }
}
