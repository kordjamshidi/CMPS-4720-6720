/**
  * Created by jaelle on 2/28/17.
  */

import scala.collection.mutable.ListBuffer
import scala.util.Random;

class ATCStudentReader(filename:String) {
  var students = ListBuffer[Student]()

  val bufferedSource = io.Source.fromFile(filename)
  var firstLine = true

  for (student <- bufferedSource.getLines()) {
    if (firstLine) {
      firstLine = false
    }
    else {
      students = students += new Student(student)
    }
  }
}
