import edu.illinois.cs.cogcomp.saul.datamodel.DataModel

object ATCStudentAllDataModel extends DataModel {
  val studentAllData = node[StudentAllData]

  val allFeatures = property(studentAllData) {
    x: StudentAllData =>
      x.columns.toList
  }

  val studentAllDataLabel = property(studentAllData)("0","1") {
    x:StudentAllData => x.label
  }
}
