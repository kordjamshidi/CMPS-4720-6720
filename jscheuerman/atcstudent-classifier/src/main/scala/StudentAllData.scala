import scala.collection.mutable.ListBuffer

/**
  * Created by jscheuerman on 3/28/2017.
  */
class StudentAllData(student: String) {

  var columns = new ListBuffer[Double]()
  var i = 0;
  var label = ""

  student.split(",").foreach(column => {
    //skip first column. This is the ID
    //skip 2nd column, irrelevant
    //Second column is the label
    if (i == 2) {
      label = column
    }
    if (i > 2) {
      columns += cleanString(column).toDouble

    }
    i = i + 1
  })

  def cleanString(str: String): String = if (str.contains("NULL") || str == "") """0""" else str

}
