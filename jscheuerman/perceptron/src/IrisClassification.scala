/**
  * Name: Jaelle Scheuerman
  * Course: CMPS 6720
  * Assignment: Perceptron programming assignment
  */

import scala.io.Source
import scala.collection.mutable.ArrayBuffer

object IrisClassification {
  def main(args: Array[String]): Unit = {
    val filename = "iris.data"
    val numInputs = 4
    val learningRate = 0.5

    //create perceptron for each output label
    val setosa = new Perceptron("Iris-setosa",numInputs)
    val versicolor = new Perceptron("Iris-versicolor",numInputs)
    val virginica = new Perceptron("Iris-virginica",numInputs)

    //array of perceptrons
    val perceptrons = Array[Perceptron](setosa,versicolor,virginica)
    var data = ArrayBuffer[Array[String]]()

    //load file input into data structure
    //label is expected to be the last column
    for (line <- Source.fromFile(filename).getLines) {
      data += line.split(",").map(_.trim)
    }

    train(perceptrons,data,learningRate)
    perceptrons.foreach(perceptron => {
      println(perceptron.category + " perceptron weights: " + perceptron.weights.deep.mkString(","))
    })
    predict(perceptrons,data)
  }

  def train(perceptrons: Array[Perceptron], data: ArrayBuffer[Array[String]], learningRate: Double): Unit = {
    var learning = true
    var count = 0

    while (learning && count < 100) {
      learning = false
      count = count + 1

      for (row <- data) {
        val category = row.last.toString
        val inputs = row.dropRight(1).map(input => input.toDouble)

        perceptrons.foreach(perceptron => {
          val output = perceptron.getNet(inputs)

          perceptron.category match {
              //labels match
            case c if c == category => {
              // this perceptron should be positive
              output.signum match {
                case -1 | 0 => {
                  perceptron.updateWeights(inputs, learningRate, 1)
                  learning = true
                }
                case 1 => null
              }
            }
            case c if c != category => {
              // this perceptron should be negative
              output.signum match {
                case -1 | 0 => null
                case 1 => {
                  perceptron.updateWeights(inputs, learningRate, -1)
                  learning = true
                }
              }
            }
          }
        })
      }
    }
  }

  def predict(perceptrons: Array[Perceptron], data: ArrayBuffer[Array[String]]): Unit = {
    for (row <- data) {
      val category = row.last.toString()
      val inputs = row.dropRight(1).map(input => input.toDouble)

      perceptrons.foreach(perceptron => {
        val output = perceptron.getNet(inputs).signum
        output.signum match {
          case o if (o == -1 || o == 0) => println(category + " is predicted to not match " + perceptron.category)
          case o if (o == 1) => println(category + " is predicted to match " + perceptron.category)
        }
      })
    }
  }

  def getLabel(category: String): Int = {
    category match {
      case "Iris-setosa" => 1
      case "Iris-versicolor" => 2
      case "Iris-virginica" => 3
    }
  }


}
