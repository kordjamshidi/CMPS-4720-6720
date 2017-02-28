/**
  * Created by jaelle on 2/28/17.
  */
import scala.collection.mutable.ListBuffer

class Student(student: String) {
  var columns = new ListBuffer[String]()
  student.split(",").foreach(column => columns += column)
}
