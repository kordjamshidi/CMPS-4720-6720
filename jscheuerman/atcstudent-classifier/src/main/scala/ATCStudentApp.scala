/**
  * Created by jscheuerman on 2/24/2017.
  */

import edu.illinois.cs.cogcomp.saul.util.Logging

object ATCStudentApp extends Logging {
  val studentData = new ATCStudentReader("example-data.csv")
  val testData = studentData.test

  def main(args: Array[String]): Unit = {

    ATCStudentDataModel.student populate studentData.training
    ATCStudentClassifier.forget()
    ATCStudentClassifier.learn(30)
    ATCStudentClassifier.test(studentData.test)

  }
}