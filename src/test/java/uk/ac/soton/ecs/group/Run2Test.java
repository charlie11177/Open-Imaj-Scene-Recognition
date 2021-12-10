package uk.ac.soton.ecs.group;

import static org.junit.Assert.*;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.junit.Test;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.util.Map;

/**
 * Unit test for simple Run2.
 */
public class Run2Test {
    /**
     * Rigourous Test :-)
     */
	@Test
    public void testApp() throws FileSystemException {
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 50, 0, 50);

        HardAssigner<DoubleFV, float[], IntFloatPair> assigner = Run2.trainQuantiser(training);

        FeatureExtractor<SparseIntFV, FImage> extractor = new Run2.POWExtractor(assigner);

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splits.getTrainingDataset());

        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result);
        assertTrue("Testing if accuracy above 1/15", result.getMatrix().getAccuracy() > (1/15));
    }
}
