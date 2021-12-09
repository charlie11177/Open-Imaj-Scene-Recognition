package uk.ac.soton.ecs.group;


import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.*;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

public class Run2 {
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> testing = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);

        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(training);
    }


    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample) {
        List<List<float[]>> allPatches = new ArrayList<List<float[]>>();

        for (FImage image : sample) {
            List<float[]> patches = new PatchExtractor(image).extractPatches();
            allPatches.add(patches);
        }

        if (allPatches.size() > 10000)
            allPatches = allPatches.subList(0, 10000);

        FloatKMeans cluster = FloatKMeans.createKDTreeEnsemble(500);

        //TODO: Convert List<List<float[]>> to float[][]
        // Main issue is the patches: do we need them as float[], or can we convert them to float ??
        // what is the point of mean-centering a patch (an array) and normalising ????

        // FloatArrayBackedDataSource needs a float[][]
        DataSource<float[]> dataSource = new FloatArrayBackedDataSource(allPatches);
        FloatCentroidsResult result = cluster.cluster(dataSource);

        return result.defaultHardAssigner();

    }
}
