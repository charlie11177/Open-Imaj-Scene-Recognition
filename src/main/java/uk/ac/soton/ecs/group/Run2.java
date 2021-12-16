package uk.ac.soton.ecs.group;


import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.math3.geometry.euclidean.twod.Euclidean2D;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.vfs2.FileObject;
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
import org.openimaj.io.IOUtils;
import org.openimaj.knn.ObjectNearestNeighbours;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.ml.clustering.kmeans.KMeansConfiguration;
import org.openimaj.util.comparator.DistanceComparator;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Run2 {
    public static void main( String[] args ) throws IOException {
        //Collecting the training and testing set from the coursework webpage
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        //Removing the empty training group created by the zip
        training.remove("training");
        VFSListDataset<FImage> testing = new VFSListDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);
        String path = "./src/main/java/uk/ac/soton/ecs/group/";

        HardAssigner<DoubleFV, float[], IntFloatPair> assigner = trainQuantiser(training);

        FeatureExtractor<SparseIntFV, FImage> extractor = new POWExtractor(assigner);

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(training);

        PrintWriter output = new PrintWriter(path+"run2.txt");
        int index = 0;
        //Cycles through all images in the testing set. Gets the filename, attempts to classify the image and then writes the output to the console and the file "run2.txt"
        for(FImage image : testing){
            String filename = testing.getID(index).replace("testing/", "");
            String guess = ann.classify(image).getPredictedClasses().toArray(new String[0])[0];
            System.out.println(filename + " " + guess);
            output.println(filename + " " + guess);
            index++;
        }
        output.close();
    }


    /**
     * HardAssigner which takes a sample of images and from each image extracts a List of DoubleFV which represent 8x8 patches across the image.
     * It then takes a sample of this list and uses FeatureVectorKMeans to cluster them into 500 different clusters
     * @param sample The sample Dataset of images to train on
     * @return A HardAssigner used to distinguish which cluster a DoubleFV (patch) belongs to
     */
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

        return result.defaultHardAssigner();
    }

    static class POWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        HardAssigner<DoubleFV, float[], IntFloatPair> assigner;

        public POWExtractor(HardAssigner<DoubleFV, float[], IntFloatPair> assigner){
            this.assigner = assigner;
        }

        /**
         * Takes an image and using a HardAssigner, creates a BagOfVisual words representation to distinguish how many local features come from each cluster in the assigner
         * It then aggregates these values to create a vector
         * @param image Image to extract feature from
         * @return SparseIntFV representing the BagOfVisual words created from patches
         */
        public SparseIntFV extractFeature(FImage image) {
            BagOfVisualWords<DoubleFV> bovw = new BagOfVisualWords<DoubleFV>(assigner);
            return bovw.aggregateVectorsRaw(new PatchExtractor(image).extractFeatureVectors());

        }
    }
}
