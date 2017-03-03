/**
  * Created by jscheuerman on 2/24/2017.
  */

import edu.illinois.cs.cogcomp.saul.util.Logging

object ATCStudentApp extends Logging {
  val studentData = new ATCStudentReader("example.csv")

  def main(args: Array[String]): Unit = {
    val (training,test) = studentData.students.splitAt(80)

    ATCStudentDataModel.student.populate(training)
    ATCStudentClassifier.forget()
    ATCStudentClassifier.learn(50)
    ATCStudentClassifier.test(test)

  }
}