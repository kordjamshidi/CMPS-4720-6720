import scala.collection.mutable.ListBuffer

class ATCStudentReader(filename:String) {
  var students = ListBuffer[Student]()
  var normalized = ListBuffer[Student]()

  val bufferedSource = io.Source.fromFile(filename)
  var firstLine = true
  var minValue = 10000.0
  var maxValue = 0.0
  var labels = ListBuffer[String]()

  for (student <- bufferedSource.getLines()) {
    if (firstLine) {
      labels = makeLabels(student)
      firstLine = false
    }
    else {
      val newStudent = new Student(student)
      newStudent.columns.foreach(column => {
        if (minValue < column) {
          minValue = column
        }
        if (maxValue > column) {
          maxValue = column
        }
      })
      students = students += newStudent
    }
  }
  val range = maxValue - minValue
  normalized = students
  normalized.foreach(student => {
    var i = 0
    while (i < student.columns.length) {
      student.columns.update(i, student.columns(i) / range)
      i = i + 1
    }
  })

  def makeLabels(student: String): ListBuffer[String] = {
    labels = ListBuffer[String]()
    var i = 0;
    val columns = student.split(",")

    columns.foreach(column => {
      //ignore first three columns
      if (i > 2) {
        labels += column
      }
      i = i + 1
    })
    labels
  }
}