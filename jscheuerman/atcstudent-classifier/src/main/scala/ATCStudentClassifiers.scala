/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import ATCStudentDataModel._
import edu.illinois.cs.cogcomp.lbjava.learn.{Sigmoid, SparseNetworkLearner}

object ATCStudentClassifiers {

  object ATCStudentClassifier extends Learnable(student) {
    def label = studentLabel;
    override lazy val classifier = new LassoPerceptron();
    def errors = classifier.errors
    override def feature = using(afqt,dotGroup,nbackGroup,ospanGroup,sspanGroup,age,gender,gaming,eyecorrected,handed,eyedom)

  }
  object ATCStudentNetwork extends Learnable(student) {
    def label = studentLabel;
    override lazy val classifier = new SparseNetworkLearner();
    override def feature = using(afqt,dotGroup,nbackGroup,ospanGroup,sspanGroup,age,gender,gaming,eyecorrected,handed,eyedom)
  }
}
