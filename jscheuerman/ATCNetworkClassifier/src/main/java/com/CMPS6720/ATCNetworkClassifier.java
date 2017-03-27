package com.CMPS6720;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class ATCNetworkClassifier
{
    private static Logger log = LoggerFactory.getLogger(ATCNetworkClassifier.class);

    public static void main( String[] args ) throws Exception
    {
        int numLinesToSkip = 1;
        String delimeter = ",";

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimeter);
        File file = new File("example.csv");

        recordReader.initialize(new FileSplit(file));
        int labelIndex = 0;
        int numClasses = 2;
        int batchSize = 80;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);

        int numInputs = 200;
        int outputNum = 2;
        int iterations = 50;
        long seed = 6;
        log.info("Build model....");

        int i = 0;
        int maxIterations = 50;

        //do {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(1000)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.UNIFORM)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                            .build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                            .activation(Activation.SOFTSIGN)
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                            .activation(Activation.SIGMOID)
                            .nIn(3).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(1));

            model.fit(trainingData);

            //evaluate the model on the test set
            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatureMatrix());
            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());
        //} while (i < maxIterations);
    }
}
