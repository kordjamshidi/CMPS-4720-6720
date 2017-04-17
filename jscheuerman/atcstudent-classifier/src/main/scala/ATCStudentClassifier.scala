/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.lbjava.learn.SparsePerceptron
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import ATCStudentAllDataModel._
import ATCStudentDataModel._

object ATCStudentClassifiers {

  object ATCStudentClassifier extends Learnable(studentAllData) {
    def label = studentAllDataLabel;
    override lazy val classifier = new SparsePerceptron();
    override def feature = using(allFeatures)
  }
  object ATCStudentLasso extends Learnable(studentAllData) {
    def label = studentAllDataLabel;
    override lazy val classifier = new LassoPerceptron();
    override def feature = using(allFeatures)
  }
  object ATCStudentLassoAbridged extends Learnable(student) {
    def label = studentLabel;
    override lazy val classifier = new LassoPerceptron();
    override def feature = using(afqt,dotGroup,nbackGroup,ospanGroup,sspanGroup,age,gender,gaming,eyecorrected,handed,eyedom)
  }
}
