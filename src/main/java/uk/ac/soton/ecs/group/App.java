package uk.ac.soton.ecs.group;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.*;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

public class App {
    /**
     *
     * @param args
     */
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> dataset;

        if(args.length == 1){
            dataset = new VFSGroupDataset<FImage>("zip:"+args[0], ImageUtilities.FIMAGE_READER);

        }else{
            throw new IllegalArgumentException("Incorrect arguments!");
        }//Use this when complete
    }
}
