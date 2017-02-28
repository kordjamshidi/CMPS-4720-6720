/**
  * Created by jscheuerman on 2/24/2017.
  */

import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector, _}

object ATCStudentClassifier {
  def main(args: Array[String]): Unit = {
    val studentsTrain = new ATCStudentReader("example-data.csv").students

    studentsTrain.foreach(student => {
      student.columns.foreach(column => println(column))
    })
  }
}