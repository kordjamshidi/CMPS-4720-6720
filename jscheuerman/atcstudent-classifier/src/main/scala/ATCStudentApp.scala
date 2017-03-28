import ATCStudentClassifiers._
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.lbjava.learn.SparseWeightVector

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import java.io._
import java.util

import breeze.numerics.abs
import au.com.bytecode.opencsv.CSVWriter

object ATCStudentApp extends Logging {
  val allStudentData = new ATCStudentAllDataReader("P:\\ATC Data\\air_traffic_control_school_noelle\\Jaelle\\students_clean.csv")
  val abridgedStudentData = new ATCStudentReader("P:\\ATC Data\\air_traffic_control_school_noelle\\Jaelle\\students_clean.csv")
  val allStudentTestData = new ATCStudentAllDataReader("P:\\ATC Data\\air_traffic_control_school_noelle\\Jaelle\\students_clean_test.csv")
  val abridgedStudentTestData = new ATCStudentReader("P:\\ATC Data\\air_traffic_control_school_noelle\\Jaelle\\students_clean_test.csv")
  val allStudentData = new ATCStudentAllDataReader("example.csv")
  val abridgedStudentData = new ATCStudentReader("example.csv")
  val allStudentTestData = new ATCStudentAllDataReader("example.csv")
  val abridgedStudentTestData = new ATCStudentReader("example.csv")

  def main(args: Array[String]): Unit = {


    val (training, validation) = allStudentData.normalized.splitAt(70)
    //validateModelAllData(training,validation)

    //Use full data set above
    //val (abridgedTraining, abridgedValidation) = abridgedStudentData.normalized.splitAt(70)
    //validateModelAbridged(abridgedTraining, abridgedValidation)

    ATCStudentAllDataModel.studentAllData.populate(training)

    //Best Found from above:
    var regularization = 0.08
    var numIterations = 32

    ATCStudentLasso.forget()
    ATCStudentLasso.classifier.setRegularization(regularization)
    ATCStudentLasso.learn(numIterations)
    outputWeightsWithLabels(0,allStudentData.labels,ATCStudentLasso.classifier.getWeightVector())
    ATCStudentLasso.test(validation)
    ATCStudentLasso.test(allStudentTestData.normalized)

  }

  def validateModelAllData(training: ListBuffer[StudentAllData], validation: ListBuffer[StudentAllData]): Unit = {
    ATCStudentAllDataModel.studentAllData.populate(training)

    val file = new BufferedWriter(new FileWriter("accuracy_alldata.csv"))
    val writer = new CSVWriter(file)
    var outputResults = ListBuffer[Array[String]]()
    var row = Array("Model type","Regularization","Iteration","Accuracy","Precision (Label: 0)","Precision (Label: 1)","Recall (Label: 0)","Recall (Label: 1)")
    outputResults += row

    val regularizationTerms = List(0.01,0.02,0.04,0.08,0.1,0.3,0.5,1.25,2.5,5,10)
    regularizationTerms.foreach(regularizationTerm => {
      ATCStudentLasso.forget()
      System.out.println("***********Testing regularization value " + regularizationTerm + "***********")
      ATCStudentLasso.classifier.setRegularization(regularizationTerm)

      var i = 0
      var maxIterations = 50

      while (i < maxIterations) {
        ATCStudentLasso.learn(1)
        val results = ATCStudentLasso.test(validation)

        var numCorrect = 0.0
        var totalLabel = 0.0
        var label1Precision = 0.0
        var label2Precision = 0.0
        var label1Recall = 0.0
        var label2Recall = 0.0

        results.perLabel.foreach(result => {
          numCorrect += result.correctSize
          totalLabel += result.labeledSize
          if (result.label == "0") {
            label1Precision = result.precision
            label1Recall = result.recall
          }
          else if (result.label == "1") {
            label2Precision = result.precision
            label2Recall = result.recall
          }
        })

        i = i + 1
        val accuracy = numCorrect/totalLabel
        row = Array("All data",regularizationTerm.toString(),i.toString(),accuracy.toString(),label1Precision.toString(),label2Precision.toString(),label1Recall.toString(),label2Recall.toString())
        outputResults += row
      }
    })

    writer.writeAll(outputResults.toList)
    file.close()
  }

  def validateModelAbridged(training: ListBuffer[Student], validation: ListBuffer[Student]): Unit = {
    ATCStudentDataModel.student.populate(training)

    val file = new BufferedWriter(new FileWriter("accuracy_abridgeddata.csv"))
    val writer = new CSVWriter(file)
    var outputResults = ListBuffer[Array[String]]()
    var row = Array("Model type","Regularization","Iteration","Accuracy","Precision (Label: 0)","Precision (Label: 1)","Recall (Label: 0)","Recall (Label: 1)")
    outputResults += row

    val regularizationTerms = List(0.01,0.02,0.04,0.08,0.1,0.3,0.5,1.25,2.5,5,10)
    regularizationTerms.foreach(regularizationTerm => {
      ATCStudentLassoAbridged.forget()
      System.out.println("***********Testing regularization value " + regularizationTerm + "***********")
      ATCStudentLassoAbridged.classifier.setRegularization(regularizationTerm)

      var i = 0
      var maxIterations = 50

      while (i < maxIterations) {
        ATCStudentLasso.learn(1)
        val results = ATCStudentLassoAbridged.test(validation)

        var numCorrect = 0.0
        var totalLabel = 0.0
        var label1Precision = 0.0;
        var label2Precision = 0.0;
        var label1Recall = 0.0;
        var label2Recall = 0.0;

        results.perLabel.foreach(result => {
          numCorrect += result.correctSize
          totalLabel += result.labeledSize
          if (result.label == "0") {
            label1Precision = result.precision
            label1Recall = result.recall
          }
          else if (result.label == "1") {
            label2Precision = result.precision
            label2Recall = result.recall
          }
        })

        i = i + 1
        val accuracy = numCorrect/totalLabel
        row = Array("All data",regularizationTerm.toString(),i.toString(),accuracy.toString(),label1Precision.toString(),label2Precision.toString(),label1Recall.toString(),label2Recall.toString())
        outputResults += row
      }
    })

    writer.writeAll(outputResults.toList)
    file.close()
  }

  def outputWeightsWithLabels(iteration: Integer, labels: ListBuffer[String], weights: SparseWeightVector) {
    var outputResults = ListBuffer[Array[String]]()

    val file = new BufferedWriter(new FileWriter("weights_" + iteration.toString() + ".csv"))
    val writer = new CSVWriter(file)
    var row = Array("Feature Name:","Weight Value:")
    outputResults += row
    var i = 0;
    labels.foreach(label => {
      row = Array(label,abs(weights.getWeight(i)).toString())
      outputResults += row
      i = i + 1
    })

    writer.writeAll(outputResults.toList)
    file.close()
  }
}