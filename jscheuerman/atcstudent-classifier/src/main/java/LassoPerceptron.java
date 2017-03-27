/**
 * Created by jscheuerman on 3/17/2017.
 */

import edu.illinois.cs.cogcomp.lbjava.learn.Learner;
import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseWeightVector;
import java.util.ArrayList;
import java.lang.Math;

import java.io.PrintStream;

public class LassoPerceptron extends SparsePerceptron  {

    //double regularizationTerm = 0.1;
    public ArrayList<Double> errors = new ArrayList<Double>();

    /*public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
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
    }*/

    //Copied from linear threshhold unit
    public void learn(int[] exampleFeatures, double[] exampleValues, int[] exampleLabels, double[] labelValues) {
        assert exampleLabels.length == 1 : "Example must have a single label.";

        assert exampleLabels[0] == 0 || exampleLabels[0] == 1 : "Example has unallowed label value.";

        boolean label = exampleLabels[0] == 1;
        double s = this.score(exampleFeatures, exampleValues);
        double error = s - exampleLabels[0];
        errors.add(Math.abs(error));
        if(this.shouldPromote(label, s, this.threshold, this.positiveThickness)) {
            this.promote(exampleFeatures, exampleValues, this.computeLearningRate(exampleFeatures, exampleValues, s, label));
        }

        if(this.shouldDemote(label, s, this.threshold, this.negativeThickness)) {
            this.demote(exampleFeatures, exampleValues, this.computeLearningRate(exampleFeatures, exampleValues, s, label));
        }

    }

    //Copied from SparsePerceptron for reference
    public double score(int[] exampleFeatures, double[] exampleValues) {
        double s = this.weightVector.dot(exampleFeatures, exampleValues, this.initialWeight) + this.bias;
        return s;
    }

    public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, rate, this.initialWeight);
        this.bias += rate;
    }

    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, -rate, this.initialWeight);
        this.bias -= rate;
    }
}
