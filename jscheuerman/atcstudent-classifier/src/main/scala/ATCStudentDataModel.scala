/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel

object ATCStudentDataModel extends DataModel {
  val student = node[Student]

  val afqt = property(student) {
    x: Student => x.getAFQT
  }

  val age = property(student) {
    x: Student => x.getAge
  }

  val gaming = property(student) {
    x: Student => x.getGamingExpertise
  }

  val studentLabel = property(student)("0","1") {
    x: Student => x.getLabel
  }
}
