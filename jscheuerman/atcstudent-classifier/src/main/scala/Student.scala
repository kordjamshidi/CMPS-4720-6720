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

  def getLabel(): String = {
    label
  }

  def getAFQT(): Double = {
    columns(Col_AFQT)
  }

  def getDOTGroup(): List[Double] = {
    var dot = ListBuffer[Double]()
    dot += columns(Col_Mean_Experiment_RTcr)
    dot += columns(Col_Mean_Experiment_RTincr)
    dot += columns(Col_Experiment_pc)

    dot.toList
  }

  def getMeanExpermentRTcr(): Double = {
    columns(Col_Mean_Experiment_RTcr)
  }

  def getMeanExperimentRTincr(): Double = {
    columns(Col_Mean_Experiment_RTincr)
  }

  def getExperimentPc(): Double = {
    columns(Col_Experiment_pc)
  }

  def getNbackGroup(): List[Double] = {

    var nback = ListBuffer[Double]()
    nback += columns(Col_Nback_LibCorr_pc)
    nback += columns(Col_Mean_RT)
    nback += columns(Col_Mean_lag0_RTcorr)
    nback += columns(Col_Mean_lag1_RTcorr)
    nback += columns(Col_Mean_lag2_RTcorr)
    nback += columns(Col_Mean_lag3_RTcorr)
    nback += columns(Col_Lag0_pc)
    nback += columns(Col_Lag1_pc)
    nback += columns(Col_Lag2_pc)
    nback += columns(Col_Lag3_pc)
    nback.toList
  }

  def getNbackLibCorrPc(): Double = {
    columns(Col_Nback_LibCorr_pc)
  }

  def getMeanRT(): Double = {
    columns(Col_Mean_RT)
  }

  def getMeanLag0RTcorr(): Double = {
    columns(Col_Mean_lag0_RTcorr)
  }

  def getMeanLag1RTcorr(): Double = {
    columns(Col_Mean_lag1_RTcorr)
  }

  def getMeanLag2RTcorr(): Double = {
    columns(Col_Mean_lag2_RTcorr)
  }

  def getMeanLag3RTcorr(): Double = {
    columns(Col_Mean_lag3_RTcorr)
  }

  def getLag0Pc(): Double = {
    columns(Col_Lag0_pc)
  }

  def getLag1Pc(): Double = {
    columns(Col_Lag1_pc)
  }

  def getLag2Pc(): Double = {
    columns(Col_Lag2_pc)
  }

  def getLag3Pc(): Double = {
    columns(Col_Lag3_pc)
  }


  def getOspanGroup(): List[Double] = {
    var ospan = ListBuffer[Double]()
    ospan += columns(Col_OspanAbsoluteScore)
    ospan += columns(Col_OspanPartialScore)

    ospan.toList
  }

  def getOspanAbsoluteScore(): Double = {
    columns(Col_OspanAbsoluteScore)
  }

  def getOspanPartialScore(): Double = {
    columns(Col_OspanPartialScore)
  }

  def getSspanGroup(): List[Double] = {
    var sspan = ListBuffer[Double]()
    sspan += columns(Col_SspanAbsoluteScore)
    sspan += columns(Col_SspanPartialScore)

    sspan.toList
  }

  def getSspanAbsoluteScore(): Double = {
    columns(Col_SspanAbsoluteScore)
  }

  def getSspanPartialScore(): Double = {
    columns(Col_SspanPartialScore)
  }

  def getAge(): Double = {
    columns(Col_Age)
  }

  def getGender(): Double = {
    columns(Col_Gender)
  }

  def getGamingExpertise(): Double = {
    columns(Col_GamingExpertise)
  }

  def getEyeCorrected(): Double = {
    columns(Col_EyeCorrected)
  }

  def getHanded(): Double = {
    columns(Col_Handed)
  }

  def getEyeDom(): Double = {
    columns(Col_EyeDom)
  }

  def cleanString(str: String): String = if (str.contains("NULL") || str == "") """0""" else str
}
