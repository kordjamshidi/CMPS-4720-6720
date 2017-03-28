import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron;

/**
 * Created by jscheuerman on 3/28/2017.
 */
public class LassoPerceptron extends SparsePerceptron {
    private double regularization = 0.1;

    public void setRegularization(double value) {
        regularization = value;
    }
    public double score(int[] exampleFeatures, double[] exampleValues) {
        double l1Norm = 0.0;

        //take the l1-norm
        for (int i=0;i<this.weightVector.size();i++) {
            l1Norm += this.weightVector.getWeight(i);
        }

        //multiply by regularization value to get regularization term
        double regularizationTerm = l1Norm * regularization;
        return this.weightVector.dot(exampleFeatures, exampleValues, this.initialWeight) + regularizationTerm;
    }
}
