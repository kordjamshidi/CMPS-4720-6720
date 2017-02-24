/**
  * Created by jscheuerman on 2/24/2017.
  */

import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector, _}

object ATCStudentClassifier {
  def main(args: Array[String]): Unit = {
    val filename = "example-data.csv"
    //first row is feature names
    //data will all be double, ints or null

    var data = ArrayBuffer[Array[String]]()
    var input = DenseVector[Double]()
    var inputSize = 0
    var counter = 0
    var i =0;

    for (line <- Source.fromFile(filename).getLines) {
      if (counter == 0) {
        //first column is column names
        //make input size the size of the first row
        inputSize = line.split(',').length
        input = DenseVector.zeros[Double](inputSize)
      }
      else {
        //data += line.split(",").map(_.trim)
        line.foreach(column => {
          input(i) = column
          i = i+1
        })
        i = 0
      }
      counter = counter+1
      input.foreach(inputColumn => {
        println(inputColumn)
      })
    }

    println("Total columns:" + inputSize)
  }

}