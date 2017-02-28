/**
  * Created by jaelle on 2/28/17.
  */

import scala.collection.mutable.ListBuffer

class ATCStudentReader(filename:String) {
  var students = ListBuffer[Student]()
  val bufferedSource = io.Source.fromFile(filename)

  for (student <- bufferedSource.getLines()) {
    students = students += new Student(student)
  }
}
