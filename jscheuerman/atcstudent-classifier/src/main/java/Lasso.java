import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.Learner.Parameters;

public class Lasso extends SparsePerceptron {
    private double regularization = 0.1;

    public void setRegularization(double value) {
        this.regularization = value;
    }

    public void learn(int[] exampleFeatures, double[] exampleValues, int[] exampleLabels, double[] labelValues) {
        assert exampleLabels.length == 1 : "Example must have a single label.";

        assert exampleLabels[0] == 0 || exampleLabels[0] == 1 : "Example has unallowed label value.";

        boolean label = exampleLabels[0] == 1;
        double score = this.score(exampleFeatures,exampleValues);
        double newWeight = 0.0;

        //need to initialize weights on first pass
        if (this.weightVector.size() == 0) {
            if (this.shouldPromote(label,score,threshold,positiveThickness)) {
                this.promote(exampleFeatures,exampleValues,learningRate);
            }
            if (this.shouldDemote(label,score,threshold,negativeThickness)) {
                this.promote(exampleFeatures,exampleValues,learningRate);

            }
        }
        else {

            for (int i = 0; i < this.weightVector.size(); i++) {
                //calculate score without the feature
                double scoreWithoutFeature = score - this.weightVector.getWeight(i) * exampleValues[i] - bias / this.weightVector.size();
                double difference = Math.abs(score - scoreWithoutFeature);

                //is difference in scoreWithoutFeature i negligible?
                if (difference < -regularization / 2.0 || difference > regularization / 2.0) {
                    // should we promote value with this score?
                    if (this.shouldPromote(label,scoreWithoutFeature,threshold,(regularization / 2.0))) {
                        this.weightVector.setWeight(i, this.weightVector.getWeight(i) + exampleValues[i] * +learningRate);
                        this.bias += learningRate / this.weightVector.size();
                    }
                    // should we demote value with this score?
                    if (this.shouldDemote(label,scoreWithoutFeature,threshold,(regularization / 2.0))) {
                        this.weightVector.setWeight(i, this.weightVector.getWeight(i) + exampleValues[i] * -learningRate);
                        this.bias -= learningRate / this.weightVector.size();
                    }
                } else {
                    //ignore this feature
                    this.weightVector.getWeight(i, 0.0);
                }

            }
        }
    }

    public boolean shouldPromote(boolean label, double s, double threshold, double positiveThickness) {
        return label && s < threshold + positiveThickness;
    }

    public boolean shouldDemote(boolean label, double s, double threshold, double negativeThickness) {
        return !label && s >= threshold - negativeThickness;
    }

    /*
    public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, rate, this.initialWeight);
        this.bias += rate;
    }

    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {
        this.weightVector.scaledAdd(exampleFeatures, exampleValues, -rate, this.initialWeight);
        this.bias -= rate;
    }
    */
}