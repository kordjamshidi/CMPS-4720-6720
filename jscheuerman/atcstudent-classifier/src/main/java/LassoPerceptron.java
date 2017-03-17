/**
 * Created by jscheuerman on 3/17/2017.
 */

import edu.illinois.cs.cogcomp.lbjava.learn.Learner;
import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseWeightVector;

import java.io.PrintStream;

public class LassoPerceptron extends SparsePerceptron  {

    double regularizationTerm = 0.1;

    public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, rate, this.initialWeight);
        for(int i = 0; i < exampleFeatures.length; ++i) {
            double w = this.weightVector.getWeight(i, 0.0D) * regularizationTerm;
            this.weightVector.setWeight(i, w, 0.0D);
        }
        this.bias += rate * regularizationTerm;
    }

    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, -rate, this.initialWeight);
        for(int i = 0; i < exampleFeatures.length; ++i) {
            double w = this.weightVector.getWeight(i, 0.0D) * regularizationTerm;
            this.weightVector.setWeight(i, w, 0.0D);
        }
        this.bias -= rate * regularizationTerm;
    }

    //Copied from SparsePerceptron for reference
    /*public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, rate, this.initialWeight);
        this.bias += rate;
    }

    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, -rate, this.initialWeight);
        this.bias -= rate;
    }*/
}
