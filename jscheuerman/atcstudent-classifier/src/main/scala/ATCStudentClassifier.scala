/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.lbjava.learn.SupportVectorMachine
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import ATCStudentDataModel._

object ATCStudentClassifier extends Learnable(student) {
  def label = studentLabel;
  override lazy val classifier = new SupportVectorMachine()
  override def feature = using(afqt,age,gender,gaming)
}
