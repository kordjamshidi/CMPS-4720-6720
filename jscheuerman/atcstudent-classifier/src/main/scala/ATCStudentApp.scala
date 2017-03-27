/**
  * Created by jscheuerman on 2/24/2017.
  */

import edu.illinois.cs.cogcomp.saul.util.Logging
import ATCStudentClassifiers._

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import au.com.bytecode.opencsv.CSVWriter
import java.io._
import java.util

object ATCStudentApp extends Logging {
  val studentData = new ATCStudentReader("example.csv")

  def main(args: Array[String]): Unit = {
    val (training,validationAndTest) = studentData.students.splitAt(70)
    val (validation,test) = validationAndTest.splitAt(10)
    var i = 0
    var maxIterations = 50;
    var totalCorrect = 0.0
    var total = 0.0
    var accuracy = 0.0
    var outputResults = ListBuffer[Array[String]]()

    val file = new BufferedWriter(new FileWriter("accuracy.csv"))
    val writer = new CSVWriter(file)

    ATCStudentDataModel.student.populate(training)
    while (i<maxIterations) {
      ATCStudentClassifier.learn(1)

      val results = ATCStudentClassifier.test(validation)
      var numCorrect = 0.0
      var totalLabel = 0.0
      results.perLabel.foreach(result => {
        numCorrect += result.correctSize
        totalLabel += result.labeledSize
      })
      val accuracy = numCorrect/totalLabel*100;
      var row = Array(i.toString(),accuracy.toString())
      outputResults += row

      i = i + 1
    }
    writer.writeAll(outputResults.toList)
    file.close()

    /*ATCStudentPerceptron.forget()
    ATCStudentPerceptron.learn(80)
    ATCStudentPerceptron.test(test)*/
  }
}