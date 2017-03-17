/**
  * Created by jscheuerman on 2/24/2017.
  */

import edu.illinois.cs.cogcomp.saul.util.Logging
import ATCStudentClassifiers.{ATCStudentClassifier, ATCStudentPerceptron}

object ATCStudentApp extends Logging {
  val studentData = new ATCStudentReader("test.data")

  def main(args: Array[String]): Unit = {
    val (training,test) = studentData.students.splitAt(80)

    ATCStudentDataModel.student.populate(training)
    ATCStudentClassifier.forget()
    ATCStudentClassifier.learn(50)
    ATCStudentClassifier.test(test)


    /*ATCStudentPerceptron.forget()
    ATCStudentPerceptron.learn(80)
    ATCStudentPerceptron.test(test)*/
  }
}