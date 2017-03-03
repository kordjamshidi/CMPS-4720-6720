/**
  * Created by jaelle on 2/28/17.
  */
import scala.collection.mutable.ListBuffer

class Student(student: String) {

  val Col_Unit1_NonGrad = 2
  val Col_AFQT = 3
  val Col_Mean_Experiment_RTcr = 13
  val Col_Mean_Experiment_RTincr = 15
  val Col_Experiment_pc = 18
  val Col_Nback_LibCorr_pc = 94
  val Col_Mean_RT = 22
  val Col_Mean_lag0_RTcorr = 38
  val Col_Mean_lag1_RTcorr = 40
  val Col_Mean_lag2_RTcorr = 42
  val Col_Mean_lag3_RTcorr = 44
  val Col_Lag0_pc = 97
  val Col_Lag1_pc = 98
  val Col_Lag2_pc = 99
  val Col_Lag3_pc = 100
  val Col_OspanAbsoluteScore = 102
  val Col_OspanPartialScore = 103
  val Col_SspanAbsoluteScore = 108
  val Col_SspanPartialScore = 109
  val Col_Age = 113
  val Col_Gender = 127
  val Col_GamingExpertise = 128
  val Col_EyeCorrected = 131
  val Col_Handed = 132
  val Col_EyeDom = 133

  var columns = new ListBuffer[String]()
  student.split(",").foreach(column => columns += cleanString(column))
  columns.foreach(column => {
    if (column.contains("NULL")) {
      println(column)
    }
  })

  def getLabel(): String = {
    columns(Col_Unit1_NonGrad)
  }

  def getAFQT(): Double = {
    columns(Col_AFQT).toDouble
  }

  def getMeanExpermentRTcr(): Double = {
    columns(Col_Mean_Experiment_RTcr).toDouble
  }

  def getMeanExperimentRTincr(): Double = {
    columns(Col_Mean_Experiment_RTincr).toDouble
  }

  def getExperimentPc(): Double = {
    columns(Col_Experiment_pc).toDouble
  }

  def getNbackLibCorrPc(): Double = {
    columns(Col_Nback_LibCorr_pc).toDouble
  }

  def getMeanRT(): Double = {
    columns(Col_Mean_RT).toDouble
  }

  def getMeanLag0RTcorr(): Double = {
    columns(Col_Mean_lag0_RTcorr).toDouble
  }

  def getMeanLag1RTcorr(): Double = {
    columns(Col_Mean_lag1_RTcorr).toDouble
  }

  def getMeanLag2RTcorr(): Double = {
    columns(Col_Mean_lag2_RTcorr).toDouble
  }

  def getMeanLag3RTcorr(): Double = {
    columns(Col_Mean_lag3_RTcorr).toDouble
  }

  def getLag0Pc(): Double = {
    columns(Col_Lag0_pc).toDouble
  }

  def getLag1Pc(): Double = {
    columns(Col_Lag1_pc).toDouble
  }

  def getLag2Pc(): Double = {
    columns(Col_Lag2_pc).toDouble
  }

  def getLag3Pc(): Double = {
    columns(Col_Lag3_pc).toDouble
  }

  def getOspanAbsoluteScore(): Double = {
    columns(Col_OspanAbsoluteScore).toDouble
  }

  def getOspanPartialScore(): Double = {
    columns(Col_OspanPartialScore).toDouble
  }

  def getSspanAbsoluteScore(): Double = {
    columns(Col_SspanAbsoluteScore).toDouble
  }

  def getSspanPartialScore(): Double = {
    columns(Col_SspanPartialScore).toDouble
  }

  def getAge(): Int = {
    columns(Col_Age).toInt
  }

  def getGender(): Int = {
    columns(Col_Gender).toInt
  }

  def getGamingExpertise(): Int = {
    columns(Col_GamingExpertise).toInt
  }

  def getEyeCorrected(): Int = {
    columns(Col_EyeCorrected).toInt
  }

  def getHanded(): Int = {
    columns(Col_Handed).toInt
  }

  def getEyeDom(): Int = {
    columns(Col_EyeDom).toInt
  }

  //TODO: Do we need a better way to deal with NULL data?
  def cleanString(str: String): String = if (str.contains("NULL") || str == "") """0""" else str
}
