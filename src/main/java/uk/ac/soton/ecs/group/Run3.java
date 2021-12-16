package uk.ac.soton.ecs.group;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 74.8% Accuracy
 */
public class Run3 {
    public static void main(String[] args) throws FileSystemException, FileNotFoundException {
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        training.remove("training");
        VFSListDataset<FImage> testing = new VFSListDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);
        String path = "./src/main/java/uk/ac/soton/ecs/group/";

        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 20), pdsift);

        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);

        FeatureExtractor<DoubleFV, FImage> extractor = homogeneousKernelMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));


        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(training);

        PrintWriter output = new PrintWriter(path+"run3.txt");
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

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (FImage img : sample) {
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage image) {
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 4);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}
