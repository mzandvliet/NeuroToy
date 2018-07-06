using System;
using Unity.Collections;
using UnityEngine;

/* 
Todo:
- Fix fundamental problems, get the computation correct
    - Same amount of iterations, same numerical computations as baselines should result in ~same accuracy
- Better training progress logging (graph of average cost per batch)
- Code refactoring:
    - Separation of data structures needed for forward and backward evaluation; for training and use
    - vector/matrix notation
    - ...and a library that computes those efficiently

- One perceived problem right now is that randomly initialized networks don't seem
to correspond to uniform distribution over the output classes. I suspect this
has to do with suboptimal initialization strategy. Try Xavier or other.
 */

public class MnistTraining : MonoBehaviour {
    private Texture2D _tex;
    private int _label;

    private System.Random _random;
    
    private Network _net;
    private Network _gradientBucket; // Hack used to store average gradients for minibatch
    [SerializeField] private NeuralNetRenderer _renderer;

    int _batchesTrained;

    private void Awake() {
        Application.runInBackground = true;
        _random = new System.Random();

        Mnist.Load();

        _tex = new Texture2D(Mnist.Train.Rows, Mnist.Train.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;

        var def = new NetDefinition(
            Mnist.Train.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        _net = NetBuilder.Build(def);
        _gradientBucket = NetBuilder.Build(def);

        NetUtils.RandomGaussian(_net, _random);

        _renderer.SetTarget(_net);
    }

    private void TrainMinibatch() {
        const int numClasses = 10;
        const int batchSize = 10;
        var batch = Mnist.GetBatch(batchSize, Mnist.Train, _random);

        var target = new float[numClasses];
        var dCdO = new float[numClasses];

        float avgBatchCost = 0f;
        int correctLabels = 0;

        ZeroGradients(_gradientBucket);

        for (int i = 0; i < batch.Indices.Length; i++) {
            int lbl = Mnist.Train.Labels[batch.Indices[i]];

            // Copy image to input layer (Todo: this is a waste of time/memory)
            for (int p = 0; p < Mnist.Train.ImgDims; p++) {
                _net.Input[p] = Mnist.Train.Images[batch.Indices[i], p];
            }

            NetUtils.Forward(_net);

            int predictedLbl = NetUtils.GetMaxOutput(_net);
            Mnist.LabelToVector(lbl, target);

            if (predictedLbl == lbl) {
                correctLabels++;
            }
            //Debug.Log(outputClass + ", " + batch.Labels[i]);

            // Calculate error between output layer and target
            Mnist.Subtract(target, _net.Output, dCdO);
            float cost = Mnist.Cost(dCdO);
            avgBatchCost += cost;

            // Propagate error back
            // Calculate per-parameter gradient, store it

            NetUtils.Backward(_net, target);
            AddGradients(_net, _gradientBucket);
        }

        avgBatchCost /= (float)batchSize;
        DivideGradients(_gradientBucket, (float)batchSize);

        // Update weights and biases according to averaged gradient and learning rate
        float rate = 3.0f * (1f - 0.9f * Mathf.Clamp01(_batchesTrained / 99999f));
        NetUtils.UpdateParameters(_net, _gradientBucket, rate);

        _batchesTrained++;

        Debug.Log("Batch: " + _batchesTrained + ", Cost: " + avgBatchCost + ", Correct: " + correctLabels + "/" + batchSize + " Rate: " + rate);
        // Mnist.ToTexture(batch, batch.Labels.Length-1, _tex);
        // _label = batch.Labels[batch.Labels.Length-1];
    }

    private static void ZeroGradients(Network bucket) {
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                bucket.Layers[l].DCDZ[n] = 0f;
                for (int w = 0; w < bucket.Layers[l-1].NeuronCount; w++) {
                    bucket.Layers[l].DCDW[n, w] = 0f;
                }
            }
        }
    }

    private static void AddGradients(Network values, Network bucket) {
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                bucket.Layers[l].DCDZ[n] += values.Layers[l].DCDZ[n];
                for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
                    bucket.Layers[l].DCDW[n, w] += values.Layers[l].DCDW[n, w];
                }
            }
        }
    }

    private static void DivideGradients(Network bucket, float factor) {
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                bucket.Layers[l].DCDZ[n] /= factor;
                for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
                    bucket.Layers[l].DCDW[n, w] /= factor;
                }
            }
        }
    }

    private static void ClipGradients(Network bucket) {
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                bucket.Layers[l].DCDZ[n] = Mathf.Clamp(bucket.Layers[l].DCDZ[n], -1.0f, 1.0f);
                for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
                    bucket.Layers[l].DCDW[n, w] = Mathf.Clamp(bucket.Layers[l].DCDW[n, w], -1.0f, 1.0f); ;
                }
            }
        }
    }

    private void Update() {
        if (_batchesTrained < 30 * 6000) {
            for (int i = 0; i < 16; i++) {
                TrainMinibatch();
            }
        }
    }

    private void OnGUI() {
        GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _label);
        GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
    }
}
