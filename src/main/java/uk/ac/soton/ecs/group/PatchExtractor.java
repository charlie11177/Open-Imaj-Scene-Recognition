package uk.ac.soton.ecs.group;

import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class PatchExtractor {
    FImage image;

    public PatchExtractor(FImage image){
        this.image = image;
    }

    /**
     * This method extracts patches of 8x8 pixels sampled every 4 pixels after mean-centring and
     * normalising them.
     * @return a list of patches, each patch as a float[]
     */
    public List<float[]> extractPatches(){
        RectangleSampler rectangleSampler = new RectangleSampler(image, 4, 4, 8, 8);
        Iterator<FImage> rectangleIterator = rectangleSampler.subImageIterator(image);

        List<float[]> patches = new ArrayList<>();
        while(rectangleIterator.hasNext()){
            FImage patch = rectangleIterator.next();

            new MeanCenter().processImage(patch); // Mean-centring
            patch.normalise(); // Normalising

            patches.add(patch.getFloatPixelVector());
        }
        return patches;
    }
}
