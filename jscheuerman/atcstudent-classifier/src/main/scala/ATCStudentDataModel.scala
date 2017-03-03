/**
  * Created by jscheuerman on 3/3/2017.
  */

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel

object ATCStudentDataModel extends DataModel {
  val student = node[Student]

  val afqt = property(student) {
    x: Student => x.getAFQT
  }

  val meanexprtcr = property(student) {
    x: Student => x.getMeanExpermentRTcr
  }

  val meanexprtincr = property(student) {
    x: Student => x.getMeanExperimentRTincr
  }

  val exppc = property(student) {
    x: Student => x.getExperimentPc
  }

  val nbacklibcorrect = property(student) {
    x: Student => x.getNbackLibCorrPc
  }

  val meanrt = property(student) {
    x: Student => x.getMeanRT
  }

  val meanlag0rtcorr = property(student) {
    x: Student => x.getMeanLag0RTcorr
  }

  val meanlag1rtcorr = property(student) {
    x: Student => x.getMeanLag1RTcorr
  }

  val meanlag2rtcorr = property(student) {
    x: Student => x.getMeanLag2RTcorr
  }

  val meanlag3rtcorr = property(student) {
    x: Student => x.getMeanLag3RTcorr
  }

  val lag0 = property(student) {
    x: Student => x.getLag0Pc
  }

  val lag1 = property(student) {
    x: Student => x.getLag1Pc
  }

  val lag2 = property(student) {
    x: Student => x.getLag2Pc
  }

  val lag3 = property(student) {
    x: Student => x.getLag3Pc
  }

  val ospanabsscore = property(student) {
    x: Student => x.getOspanAbsoluteScore
  }

  val ospanpartscore = property(student) {
    x: Student => x.getOspanPartialScore
  }

  val sspanabsscore = property(student) {
    x: Student => x.getSspanAbsoluteScore
  }

  val sspanpartscore = property(student) {
    x: Student => x.getSspanPartialScore
  }

  val age = property(student) {
    x: Student => x.getAge
  }

  val gender = property(student) {
    x: Student => x.getGender
  }

  val gaming = property(student) {
    x: Student => x.getGamingExpertise
  }

  val eyecorrected = property(student) {
    x: Student => x.getEyeCorrected
  }

  val handed = property(student) {
    x: Student => x.getHanded
  }

  val eyedom = property(student) {
    x: Student => x.getEyeDom
  }
  val studentLabel = property(student)("0","1") {
    x: Student => x.getLabel
  }
}
