/**
  * Name: Jaelle Scheuerman
  * Course: CMPS 6720
  * Assignment: Perceptron programming assignment
  */
class Perceptron(c: String, numInputs: Int) {

  var weights = Array.fill[Double](numInputs)(0.0)
  val category = c

  def updateWeights(inputs: Array[Double], learningRate: Double, sign: Int): Unit = {

    val result = weights.zip(inputs).map{case(w,x) => w + learningRate * sign * x}
    weights = result
  }

  def getNet(inputs: Array[Double]): Double = {
    weights.zip(inputs).map{case(w,x) => w * x}.sum
  }

}
