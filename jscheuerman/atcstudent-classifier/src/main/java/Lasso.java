import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent;

import java.io.PrintStream;

/**
 * Created by jscheuerman on 3/17/2017.
 */
public class Lasso extends SparsePerceptron {

    /*public void learn(int[] exampleFeatures, double[] exampleValues, int[] exampleLabels, double[] labelValues) {
        assert exampleLabels.length == 1 : "Example must have a single label.";

        double labelValue = labelValues[0];
        double multiplier = this.learningRate * (labelValue - sign(this.weightVector.dot(exampleFeatures, exampleValues)));
        double regularization = (1 - this.learningRate*regularizationTerm);
        this.weightVector.scaledMultiply(exampleFeatures,exampleValues,regularization);
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, multiplier);
    }*/

    public String discreteValue(int[] exampleFeatures, double[] exampleValues) {
        if ((this.weightVector.dot(exampleFeatures, exampleValues) + this.bias) <= 0) {
            return "0";
        }
        else {
            return "1";
        }
    }

    public int sign(double value) {
        if (value < 0) {
            return -1;
        }
        return 1;
    }
    /*
    public void learn(int[] exampleFeatures, double[] exampleValues, int[] exampleLabels, double[] labelValues) {
        assert exampleLabels.length == 1 : "Example must have a single label.";

        double labelValue = labelValues[0];
        double multiplier = this.learningRate * (labelValue - this.weightVector.dot(exampleFeatures, exampleValues) - this.bias);
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, multiplier);
        this.bias += multiplier;
    }*/
}
