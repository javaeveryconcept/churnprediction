package com.ai.churnprediction.trainmodel.api.deeplearning4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class DeepLearning4JUtility {

    //Configure model and Build Neural Network
    public static MultiLayerNetwork configureModel(DataSet trainData) {
        int inputNum = trainData.getFeatures().columns();
        System.out.println("Input features: " + trainData.getFeatures());
        System.out.println("Number of features (input columns) your model expects." + inputNum);
        MultiLayerConfiguration conf = buildModelConfig(inputNum);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        return model;
    }

    /**
     * Creates a neural network with:
     *  - Input layer: size = number of input columns
     *  - Hidden layers: 32 → 16 neurons with ReLU activation
     *  - Output layer: 2 outputs (Yes/No churn) using Sigmoid + Binary Cross Entropy
     * Adam optimizer is used with learning rate 0.01.
     */
    private static MultiLayerConfiguration buildModelConfig(int inputNum) {
        return new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01)) // Adam optimizer with learning rate 0.01
                .list()
                .layer(new DenseLayer.Builder().nIn(inputNum).nOut(32).activation(Activation.RELU).build()) // Input Layer → Dense (32 neurons, ReLU)
                .layer(new DenseLayer.Builder().nOut(16).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) //Loss function: XENT (Cross-Entropy), good for binary classification.
                        .activation(Activation.SIGMOID)  // predicting churn (Yes/No = 1/0), a sigmoid gives a probability output
                        .nOut(1) // Binary output, Only 1 output neuron because you predict a single binary outcome — churn (yes=1, no=0).
                        .build())
                .build();
    }
}
