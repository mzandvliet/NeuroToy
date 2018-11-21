using System;
using Unity.Collections;
using UnityEngine;
using NNClassic.Mnist;
using Rng = Unity.Mathematics.Random;

public class MnistTraining : MonoBehaviour {
    private Texture2D _tex;
    private int _label;

    private Rng _rng;
    
    private Network _net;
    private Network _gradientBucket; // Hack used to store average gradients for minibatch
    [SerializeField] private NeuralNetRenderer _renderer;

    int _epoch;
    int _batch;
    float _trainingLoss;
    float _rate;

    private void Awake() {
        Application.runInBackground = true;
        _rng = new Rng(1234);

        DataManager.Load();

        _tex = new Texture2D(DataManager.Train.Rows, DataManager.Train.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;

        var def = new NetDefinition(
            DataManager.Train.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        _net = NetBuilder.Build(def);
        _gradientBucket = NetBuilder.Build(def);

        NetUtils.RandomGaussian(_net, ref _rng);

        _renderer.SetTarget(_net);

        // const int testImg = 7291;
        // _label = Mnist.Test.Labels[testImg];
        // Mnist.ToTexture(Mnist.Test, testImg, _tex);
    }

    private void Update() {
        if (_epoch < 30) {
            if (_batch < 6000) {
                for (int i = 0; i < 100; i++) {
                    TrainMinibatch();
                }
            } else {
                Test();
                _batch = 0;
                _epoch++;
            }
        }
    }

    private void OnGUI() {
        GUILayout.BeginVertical(GUI.skin.box);
        {
            GUILayout.Label("Epoch: " + _epoch);
            GUILayout.Label("Batch: " + _batch);
            GUILayout.Label("Train Loss: " + _trainingLoss);
            GUILayout.Label("Rate: " + _rate);
        }
        GUILayout.EndVertical();

        // GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _label);
        // GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
    }

    private void TrainMinibatch() {
        UnityEngine.Profiling.Profiler.BeginSample("TrainMiniBatch");

        const int numClasses = 10;
        const int batchSize = 10;
        var target = new float[numClasses];
        var dCdO = new float[numClasses];

        float avgTrainCost = 0f;
        int correctTrainLabels = 0;

        ZeroGradients(_gradientBucket);
        var trainBatch = DataManager.GetBatch(batchSize, DataManager.Train, ref _rng);
        for (int i = 0; i < trainBatch.Indices.Length; i++) {
            int lbl = DataManager.Train.Labels[trainBatch.Indices[i]];

            // Copy image to input layer (Todo: this is a waste of time/memory)
            UnityEngine.Profiling.Profiler.BeginSample("CopyInputs");
            
            for (int p = 0; p < DataManager.Train.ImgDims; p++) {
                _net.Input[p] = DataManager.Train.Images[trainBatch.Indices[i], p];
            }
                
            UnityEngine.Profiling.Profiler.EndSample();

            NetUtils.Forward(_net);

            int predictedLbl = NetUtils.GetMaxOutput(_net);
            NetUtils.LabelToOneHot(lbl, target);

            if (predictedLbl == lbl) {
                correctTrainLabels++;
            }
            //Debug.Log(outputClass + ", " + batch.Labels[i]);

            // Calculate error between output layer and target
            NetUtils.Subtract(target, _net.Output, dCdO);
            float cost = NetUtils.Cost(dCdO);
            avgTrainCost += cost;

            // Propagate error back
            // Calculate per-parameter gradient, store it

            NetUtils.Backward(_net, target);
            AddGradients(_net, _gradientBucket);
        }

        avgTrainCost /= (float)batchSize;

        // Update weights and biases according to averaged gradient and learning rate
        _rate = 3.0f / (float)batchSize;
        NetUtils.UpdateParameters(_net, _gradientBucket, _rate);

        _batch++;
        _trainingLoss = (float)Math.Round(avgTrainCost, 6);

        // Debug.Log(
        //     "Batch: " + _batchesTrained +
        //     ", TrainLoss: " + Math.Round(avgTrainCost, 6) +
        //     ", Rate: " + Math.Round(rate, 6));
        // Mnist.ToTexture(batch, batch.Labels.Length-1, _tex);
        // _label = batch.Labels[batch.Labels.Length-1];

        UnityEngine.Profiling.Profiler.EndSample();
    }

    private void Test() {
        UnityEngine.Profiling.Profiler.BeginSample("Test");
            
        int correctTestLabels = 0;
        for (int i = 0; i < DataManager.Test.NumImgs; i++) {
            int lbl = DataManager.Test.Labels[i];

            CopyInputs(_net, i);

            NetUtils.Forward(_net);
            int predictedLbl = NetUtils.GetMaxOutput(_net);

            if (predictedLbl == lbl) {
                correctTestLabels++;
            }
        }

        float accuracy = correctTestLabels / (float)DataManager.Test.NumImgs;
        Debug.Log("Test Accuracy: " + Math.Round(accuracy * 100f, 4) + "%");
            
        UnityEngine.Profiling.Profiler.EndSample();
    }

    private static void CopyInputs(Network net, int imgIdx) {
        for (int p = 0; p < DataManager.Test.ImgDims; p++) {
            net.Input[p] = DataManager.Test.Images[imgIdx, p];
        }
    }

    private static void ZeroGradients(Network bucket) {
        UnityEngine.Profiling.Profiler.BeginSample("Zero");
            
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                bucket.Layers[l].DCDZ[n] = 0f;

                for (int w = 0; w < bucket.Layers[l-1].NeuronCount; w++) {
                    bucket.Layers[l].DCDW[n, w] = 0f;
                }
            }
        }
            
        UnityEngine.Profiling.Profiler.EndSample();
    }

    private static void AddGradients(Network net, Network gradients) {
        UnityEngine.Profiling.Profiler.BeginSample("AddGradients");
        
        var lCount = gradients.Layers.Count;
        for (int l = 1; l < lCount; l++) {
            var dCdZGradients = gradients.Layers[l].DCDZ;
            var dCdZNet = net.Layers[l].DCDZ;
            var nCount = gradients.Layers[l].NeuronCount;

            for (int n = 0; n < nCount; n++) {
                dCdZGradients[n] += dCdZNet[n];
                
                var dCdWGradients = gradients.Layers[l].DCDW;
                var dCdWNet = net.Layers[l].DCDW;
                var wCount = gradients.Layers[l - 1].NeuronCount;

                for (int w = 0; w < wCount; w++) {
                    dCdWGradients[n, w] += dCdWNet[n, w];
                }
            }
        }
            
        UnityEngine.Profiling.Profiler.EndSample();
    }

    // private static void DivideGradients(Network bucket, float factor) {
    //     UnityEngine.Profiling.Profiler.BeginSample("DivideGradients");
            
    //     for (int l = 1; l < bucket.Layers.Count; l++) {
    //         for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
    //             bucket.Layers[l].DCDZ[n] /= factor;
    //             for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
    //                 bucket.Layers[l].DCDW[n, w] /= factor;
    //             }
    //         }
    //     }
            
    //     UnityEngine.Profiling.Profiler.EndSample();
    // }

    // private static void ClipGradients(Network bucket) {
    //     UnityEngine.Profiling.Profiler.BeginSample("ClipGradients");
            
    //     for (int l = 1; l < bucket.Layers.Count; l++) {
    //         for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
    //             bucket.Layers[l].DCDZ[n] = Mathf.Clamp(bucket.Layers[l].DCDZ[n], -1.0f, 1.0f);
    //             for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
    //                 bucket.Layers[l].DCDW[n, w] = Mathf.Clamp(bucket.Layers[l].DCDW[n, w], -1.0f, 1.0f); ;
    //             }
    //         }
    //     }
            
    //     UnityEngine.Profiling.Profiler.EndSample();
    // }
}
