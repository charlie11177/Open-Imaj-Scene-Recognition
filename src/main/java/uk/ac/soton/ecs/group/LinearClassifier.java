package uk.ac.soton.ecs.group;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class LinearClassifier extends LiblinearAnnotator {

  public LinearClassifier(FeatureExtractor extractor, Mode mode, SolverType solver, double C, double eps) {
    super(extractor, mode, solver, C, eps);
  }

  public static List<byte[]> getPatches(FImage image) {
    // Create iterator with patches as sub-images
    RectangleSampler rectangleSampler = new RectangleSampler(image, 4, 4, 8, 8);
    Iterator<FImage> rectangleIterator = rectangleSampler.subImageIterator(image);

    // List of 8x8 patches
    List<byte[]> patches = new ArrayList<>();
    while (rectangleIterator.hasNext()) {
      FImage img = rectangleIterator.next();
      new MeanCenter().processImage(img); // Mean-centring
      img.normalise(); // Normalising
      patches.add(img.toByteImage());
    }

    return patches;
  }

  public static HardAssigner<byte[],float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample){
    List<byte[]> patchVectors = new ArrayList<byte[]>();
    for(FImage img : sample){
      List<byte[]> imgPatchVectors = getPatches(img);
      if(imgPatchVectors.size() > 3000){
        imgPatchVectors = imgPatchVectors.subList(0, 3000);
      }
      patchVectors.addAll(imgPatchVectors);
    }
    ByteKMeans km = ByteKMeans.createExact(500);
    ByteCentroidsResult result = km.cluster((byte[][]) patchVectors.toArray());

    return result.defaultHardAssigner();
  }

  static class POWExtractor implements FeatureExtractor<DoubleFV, FImage> {
    HardAssigner<byte[],float[],IntFloatPair> assigner;

    public POWExtractor(HardAssigner<byte[],float[],IntFloatPair> assigner){
      this.assigner = assigner;
    }

    @Override
    public DoubleFV extractFeature(FImage image) {
      BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);
      return bovw.aggregateVectorsRaw(LinearClassifier.getPatches(image)).asDoubleFV();
    }
  }
}
