package uk.ac.soton.ecs.group;


import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.math3.geometry.euclidean.twod.Euclidean2D;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.knn.ObjectNearestNeighbours;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.ml.clustering.kmeans.KMeansConfiguration;
import org.openimaj.util.comparator.DistanceComparator;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Run2 {
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 50, 0, 50);

        VFSGroupDataset<FImage> testing = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);

        HardAssigner<DoubleFV, float[], IntFloatPair> assigner = trainQuantiser(training);

        FeatureExtractor<SparseIntFV, FImage> extractor = new POWExtractor(assigner);

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splits.getTrainingDataset());

        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage,ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }


    static HardAssigner<DoubleFV, float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample) {
        List<DoubleFV> allPatches = new ArrayList<DoubleFV>();

        for (FImage image : sample) {
            List<DoubleFV> patches = new PatchExtractor(image).extractFeatureVectors();
            allPatches.addAll(patches);
        }

        if (allPatches.size() > 10000)
            allPatches = allPatches.subList(0, 10000);

        FeatureVectorKMeans<DoubleFV> km = FeatureVectorKMeans.createExact(500, DoubleFVComparison.EUCLIDEAN);
        FeatureVectorKMeans.Result<DoubleFV> result = km.cluster(allPatches);

        //TODO: Now ended up with List<DoubleFV>
        // Ive created a method which turns an image into a list of DoubleFV hopefully we can use KMeans easier
        // what is the point of mean-centering a patch (an array) and normalising ???? No idea spec just said we might want to try it

        return result.defaultHardAssigner();
    }

    static class POWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        HardAssigner<DoubleFV, float[], IntFloatPair> assigner;

        public POWExtractor(HardAssigner<DoubleFV, float[], IntFloatPair> assigner){
            this.assigner = assigner;
        }

        @Override
        public SparseIntFV extractFeature(FImage image) {
            BagOfVisualWords<DoubleFV> bovw = new BagOfVisualWords<DoubleFV>(assigner);
            return bovw.aggregateVectorsRaw(new PatchExtractor(image).extractFeatureVectors());

        }
    }
}
