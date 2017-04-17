package com.CMPS6720;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class ATCNetworkClassifier
{
    private static Logger log = LoggerFactory.getLogger(ATCNetworkClassifier.class);
    private static boolean training = false;
    private static boolean testing = true;

    private static int labelIndex = 0;
    private static int numClasses = 2;
    private static int trainingBatchSize = 80;
    private static int testBatchSize = 25;
    private static int numInputs = 143;
    private static int outputNum = 2;
    private static long seed = 6;
    private static int minIterations = 37;
    private static int maxIterations = 38;

    //average number of iterations found to provide best fit across 10 folds
    //private static int testIterations = 50;

    //number of iterations that provides the best fit most commonly across 10 folds
    private static int testIterations = 37;

   // private static Double[] regularizationTerms = {0.00016, 0.00032, 0.00064, 0.00128, 0.00256, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2};

    //average term to be found to give best fit across 10 folds
    private static Double[] regularizationTerms = {0.1};

    //average term found to give best fit across 10 folds:
    private static Double testRegularization = 0.1;

    //most common term found to give best fit across 10 folds:
    //private static Double testRegularization = 0.00001;

    private static ArrayList<String[]> results = new ArrayList<String[]>();

    private static double bestF1 = 0.0;
    private static int bestIteration = 1;
    private static double bestRegularization = 0.000001;
    private static MultiLayerNetwork bestModel;

    public static void main( String[] args ) throws Exception
    {
        //Use 10-fold validation
        int maxFolds = 10;


        //Prepare record readers
        RecordReader allReader = new CSVRecordReader(1, ",");
        RecordReader trainValidateReader = new CSVRecordReader(1, ",");
        RecordReader testReader = new CSVRecordReader(1, ",");
        RecordReader headerReader = new CSVRecordReader(0, ",");

        //all data file for normalization purposes
        File allDataFile = new File("Jaelle\\studentsDL4J_nosetbacks_all.csv\\");
        allReader.initialize(new FileSplit(allDataFile));

        //iterator for all data
        DataSetIterator iteratorAll = new RecordReaderDataSetIterator(allReader,trainingBatchSize + testBatchSize,labelIndex,numClasses);
        DataSet allData = iteratorAll.next();

        //training and validation file
        File file = new File("Jaelle\\studentsDL4J_nosetbacks.csv\\");
        trainValidateReader.initialize(new FileSplit(file));

        //iterator for validation and training data
        DataSetIterator iteratorTrainingValidation = new RecordReaderDataSetIterator(trainValidateReader,trainingBatchSize,labelIndex,numClasses);
        DataSet trainingValidationData = iteratorTrainingValidation.next();

        //test data file
        File testFile = new File("Jaelle\\studentsDL4Jtest_nosetbacks.csv\\");
        testReader.initialize(new FileSplit(testFile));

        //iterator for test data
        DataSetIterator iteratorTest = new RecordReaderDataSetIterator(testReader,testBatchSize,labelIndex,numClasses);
        DataSet testData = iteratorTest.next();

        //Read first line of file in for tracking weights
        headerReader.initialize(new FileSplit(file));

        List columns = headerReader.next();
        //normalize data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(trainingValidationData);
        normalizer.transform(testData);

        if (training) {
            int counter = 0;
            while (counter < maxFolds) {

                //Pick random 75% to be training data, remaining 25% will be validation data
                trainingValidationData.shuffle();
                SplitTestAndTrain validateAndTrain = trainingValidationData.splitTestAndTrain(0.75);

                DataSet trainingData = validateAndTrain.getTrain();
                DataSet validateData = validateAndTrain.getTest();

                trainAndValidateModels(trainingData, validateData);

                BufferedWriter br;
                StringBuilder sb;

                if (bestModel != null) {

                    //output weights to file
                    br = new BufferedWriter(new FileWriter("bestweights_" + counter + ".csv"));
                    sb = new StringBuilder();
                    for (int i = 0; i < columns.size(); i++) {
                        sb.append(columns.get(i).toString());
                        sb.append(",");
                        sb.append(Math.abs(bestModel.params().getDouble(i)) + "\n");
                    }

                    br.write(sb.toString());
                    br.close();
                }

                //output evaluation metrics for each fold to file
                br = new BufferedWriter(new FileWriter("results_" + counter + ".csv"));
                sb = new StringBuilder();
                for (String[] outputRow : results) {
                    for (String s : outputRow) {
                        sb.append(s);
                        sb.append(",");
                    }
                    sb.append("\n");
                }

                br.write(sb.toString());
                br.close();
                counter++;
            }
        }

        if (testing) {
            testModel(trainingValidationData, testData);
            //output weights of test to file
            BufferedWriter br = new BufferedWriter(new FileWriter("bestweights_test.csv"));
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < columns.size(); i++) {
                sb.append(columns.get(i).toString());
                sb.append(",");
                sb.append(Math.abs(bestModel.params().getDouble(i)) + "\n");
            }

            br.write(sb.toString());
            br.close();
        }

    }

    static void testModel(DataSet allData, DataSet testData) throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(testIterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.ZERO) //initialize weights to 0
                .learningRate(0.1)
                .regularization(true).l1(testRegularization) //Use L1 Lasso regularization
                .list() //Use single layer so we can estimate feature importance from weights
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.L1) //Use L1 loss function
                        .activation(Activation.SIGMOID) //Use sigmoid activation function
                        .nIn(numInputs).nOut(outputNum).build())
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        model.fit(allData);

        bestModel = model;

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);

        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());

        results.clear(); //clear out the results for each validation set
        //header row for results
        String [] row = {"Model type","Regularization","Iteration","Accuracy","L0 Precision","L1 Precision","L0 Recall","L1 Recall","L0 F1","L1 F1"};
        results.add(row);
        String[] newRow = {
                "Test",
                testRegularization.toString(),
                String.valueOf(testIterations),
                Double.toString(eval.accuracy()),
                Double.toString(eval.precision(0)),
                Double.toString(eval.precision(1)),
                Double.toString(eval.recall(0)),
                Double.toString(eval.recall(1)),
                Double.toString(eval.f1(0)),
                Double.toString(eval.f1(1))};
        results.add(newRow);//output weights to file

        //output evaluation metrics for test to file
        BufferedWriter br = new BufferedWriter(new FileWriter("results_test.csv"));
        StringBuilder sb = new StringBuilder();

        for (String[] outputRow : results) {
            for (String s : outputRow) {
                sb.append(s);
                sb.append(",");
            }
            sb.append("\n");
        }

        br.write(sb.toString());
        br.close();
    }

    static void trainAndValidateModels(DataSet trainingData, DataSet validateData) {

        int i = 0;
        Integer j = 0;

        results.clear(); //clear out the results for each validation set
        //header row for results
        String [] row = {"Model type","Regularization","Iteration","Accuracy","L0 Precision","L1 Precision","L0 Recall","L1 Recall","L0 F1","L1 F1"};
        results.add(row);

        while (i < regularizationTerms.length) {
            j = minIterations;
            while (j < maxIterations) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .iterations(j)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.ZERO) //initialize weights to 0
                        .learningRate(0.1)
                        .regularization(true).l1(regularizationTerms[i]) //Use L1 Lasso regularization
                        .list() //Use single layer so we can estimate feature importance from weights
                        .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.L1) //Use L1 loss function
                                .activation(Activation.SIGMOID) //Use sigmoid activation function
                                .nIn(numInputs).nOut(outputNum).build())
                        .build();


                //run the model
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(1));

                model.fit(trainingData);

                //evaluate the model on the test set
                Evaluation eval = new Evaluation(3);

                INDArray output = model.output(validateData.getFeatureMatrix());
                eval.eval(validateData.getLabels(), output);
                log.info(eval.stats());
                double evalF1 = eval.f1(1);

                //save the best model
                if (evalF1 > bestF1) {
                    bestF1 = evalF1;
                    bestIteration = j;
                    bestRegularization = regularizationTerms[i];
                    bestModel = model;
                }

                //store results of each iteration
                String[] newRow = {
                        "Validation Data",
                        regularizationTerms[i].toString(),
                        j.toString(),
                        Double.toString(eval.accuracy()),
                        Double.toString(eval.precision(0)),
                        Double.toString(eval.precision(1)),
                        Double.toString(eval.recall(0)),
                        Double.toString(eval.recall(1)),
                        Double.toString(eval.f1(0)),
                        Double.toString(eval.f1(1))};
                results.add(newRow);
                j++;
            }
            i++;
        }
    }
}
