/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.lbjava.learn.SupportVectorMachine
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import ATCStudentDataModel._

object ATCStudentClassifier extends Learnable(student) {
  def label = studentLabel;
  override lazy val classifier = new SupportVectorMachine()
  override def feature = using(afqt,meanexprtcr,meanexprtincr,exppc,nbacklibcorrect,meanrt,meanlag0rtcorr,meanlag1rtcorr,meanlag2rtcorr,meanlag3rtcorr,lag0,lag1,lag2,lag3,ospanabsscore,ospanpartscore,sspanabsscore,sspanpartscore,age,gender,gaming,eyecorrected,handed,eyedom)
}
