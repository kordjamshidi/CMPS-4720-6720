/**
  * Created by jaelle on 2/28/17.
  */

import scala.collection.mutable.ListBuffer
import scala.util.Random;

class ATCStudentReader(filename:String) {
  var students = ListBuffer[Student]()
  var training = ListBuffer[Student]()
  var test = ListBuffer[Student]()

  val bufferedSource = io.Source.fromFile(filename)
  var firstLine = true

  for (student <- bufferedSource.getLines()) {
    //skip the first line
    if (firstLine) {
      firstLine = false
    }
    else {
      students = students += new Student(student)
    }
  }

  training = students

  //randomly choose 25 students for test set
  for (i <- 1 to 25) {
    //pick a random number between 0 and students.length
    val randomStudent = students(Random.nextInt(students.length))
    test += randomStudent
    training -= randomStudent
  }
}
