using System;
using Unity.Collections;
using UnityEngine;

/* 
Todo:
- Fix fundamental problems, get the computation correct
- Better training progress logging (graph of average cost per batch)
- Minibatch average gradient collection and update
- Code refactoring:
    - Separation of data structures needed for forward and backward evaluation; for training and use
    - vector/matrix notation
    - ...and a library that computes those efficiently

- One perceived problem right now is that randomly initialized networks don't seem
to correspond to uniform distribution over the output classes. I suspect this
has to do with suboptimal initialization strategy. Try Xavier or other.
 */

public class MnistTraining : MonoBehaviour {
    private float[] _pixels;
    private int[] _labels;
    private Texture2D _tex;

    private int _currentIndex;
    private System.Random _random;
    
    private Network _net;
    private Network _gradientBucket;
    [SerializeField] private NeuralNetRenderer _renderer;

    int _batchesTrained;

    private Batch _testBatch;

    private void Awake() {
        Application.runInBackground = true;
        _random = new System.Random();

        Mnist.Load(out _pixels, out _labels);

        _tex = new Texture2D(Mnist.Rows, Mnist.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;

        var def = new NetDefinition(
            Mnist.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        _net = NetBuilder.Build(def);
        _gradientBucket = NetBuilder.Build(def);

        // Todo: Xavier initialization
        NetUtils.Randomize(_net, _random);

        _renderer.SetTarget(_net);

        _testBatch = Mnist.GetBatch(4, _pixels, _labels, _random);
        Navigate(0);
    }

    private void TrainMinibatch() {
        const int numClasses = 10;
        const int batchSize = 10;
        var batch = Mnist.GetBatch(batchSize, _pixels, _labels, _random);

        var target = new float[numClasses];
        var dCdO = new float[numClasses];

        // Todo: have a buffer in which to store minibatch gradients for averaging
        // Needs code restructuring to make this easy to allocate

        float avgBatchCost = 0f;
        int correctLabels = 0;

        ZeroGradients(_gradientBucket);

        for (int i = 0; i < batch.Labels.Length; i++) {
            // Copy image to input layer
            for (int p = 0; p < Mnist.ImgDims; p++) {
                _net.Input[p] = batch.Images[i][p];
            }

            NetUtils.Forward(_net);

            int outputClass = NetUtils.GetMaxOutput(_net);
            Mnist.LabelToVector(batch.Labels[i], target);

            if (outputClass == batch.Labels[i]) {
                correctLabels++;
            }
            Debug.Log(outputClass + ", " + batch.Labels[i]);

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
        float rate = 3.0f;// / (1f + Mathf.Log(1f + (float)_batchesTrained, 2f));
        NetUtils.UpdateParameters(_net, _gradientBucket, rate);

        _batchesTrained++;

        Debug.Log("Batch: " + _batchesTrained + ", Cost: " + avgBatchCost + ", Correct: " + correctLabels + "/" + batchSize + " Rate: " + rate);
        //Mnist.ToTexture(batch, 0, _tex);
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

    private static void CheckGradients(Network bucket) {
        for (int l = 1; l < bucket.Layers.Count; l++) {
            for (int n = 0; n < bucket.Layers[l].NeuronCount; n++) {
                if (bucket.Layers[l].DCDZ[n] == 0f) {
                    Debug.Log("dBias Zero: " + l + ", " + n);
                }
                
                for (int w = 0; w < bucket.Layers[l - 1].NeuronCount; w++) {
                    if (bucket.Layers[l].DCDW[n, w] == 0f) {
                        Debug.Log("dWeight Zero: " + l + ", " + n + ", " + w);
                    }
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
        if (Input.GetKeyDown(KeyCode.D)) {
            Navigate(_currentIndex + 1);
        }
        if (Input.GetKeyDown(KeyCode.A)) {
            Navigate(_currentIndex - 1);
        }

        // if (_batchesTrained < 30) {
        //     TrainMinibatch();
        // }
    }

    private void Navigate(int index) {
        _currentIndex = Mathf.Clamp(index, 0, _testBatch.Labels.Length);
        Mnist.ToTexture(_testBatch, _currentIndex, _tex);
    }

    private void OnGUI() {
        GUI.Label(new Rect(0f, 0f, 280f, 32f), "Image: " + _currentIndex);
        GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _labels[_currentIndex]);
        GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
    }

    private void OnGizmos() {

    }
}
