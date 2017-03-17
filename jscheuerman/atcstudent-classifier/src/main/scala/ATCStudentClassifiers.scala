/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import ATCStudentDataModel._

object ATCStudentClassifiers {

  object ATCStudentClassifier extends Learnable(student) {
    def label = studentLabel;
    override lazy val classifier = new Lasso();
    override def feature = using(afqt,dotGroup,nbackGroup,ospanGroup,sspanGroup,age,gender,gaming,eyecorrected,handed,eyedom)
  }
  object ATCStudentPerceptron extends Learnable(student) {
    def label = studentLabel;
    override lazy val classifier = new LassoPerceptron();
    override def feature = using(afqt,dotGroup,nbackGroup,ospanGroup,sspanGroup,age,gender,gaming,eyecorrected,handed,eyedom)
  }
}
