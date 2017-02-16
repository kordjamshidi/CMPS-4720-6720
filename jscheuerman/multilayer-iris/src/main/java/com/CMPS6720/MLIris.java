package com.CMPS6720;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MLIris
{
    private static Logger log = LoggerFactory.getLogger(MLIris.class);

    public static void main(String[] args) throws Exception {

        //number of features
        int numInputs = 4;

        //total examples in iris data
        int numExamples = 150;

        //number of outputs
        int numOutputClasses = 3;

        int batchSize = 30;
        int seed = 6;
        int numEpochs = 50;
        double learningRate = 0.1;

        //Get the DataSetIterators:
        DataSetIterator irisTrainingSet = new IrisDataSetIterator(batchSize, numExamples);
        DataSetIterator irisTestingSet = new IrisDataSetIterator(batchSize, numExamples);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //include a random seed for reproducibility
                .iterations(5)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(learningRate) //specify the learning rate
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(numOutputClasses).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            model.fit(irisTrainingSet);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputClasses);
        while(irisTestingSet.hasNext()){
            DataSet next = irisTestingSet.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
