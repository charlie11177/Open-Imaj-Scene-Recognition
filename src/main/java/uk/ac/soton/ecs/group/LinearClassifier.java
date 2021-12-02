package uk.ac.soton.ecs.group;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import java.util.ArrayList;
import java.util.Iterator;

public class LinearClassifier extends LiblinearAnnotator {

  public LinearClassifier(FeatureExtractor extractor, Mode mode, SolverType solver, double C, double eps) {
    super(extractor, mode, solver, C, eps);
  }

  public ArrayList<FImage> getPatches(FImage image) {
    // Create iterator with patches as sub-images
    RectangleSampler rectangleSampler = new RectangleSampler(image, 4, 4, 8, 8);
    Iterator<FImage> rectangleIterator = rectangleSampler.subImageIterator(image);

    // List of 8x8 patches
    ArrayList<FImage> patches = new ArrayList<>();
    while (rectangleIterator.hasNext()) {
      FImage img = rectangleIterator.next();
      new MeanCenter().processImage(img); // Mean-centring
      img.normalise(); // Normalising
      patches.add(img);
    }

    return patches;
  }

}
