using System;
using Unity.Collections;
using UnityEngine;

/* 
Todo:
- Better training progress logging (graph of average cost per batch)
- Minibatch average gradient collection and update
- Code refactoring

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
    [SerializeField] private NeuralNetRenderer _renderer;

    int _batchesTrained;

    private void Awake() {
        Application.runInBackground = true;
        _random = new System.Random(1234);

        Mnist.Load(out _pixels, out _labels);

        _tex = new Texture2D(Mnist.Rows, Mnist.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;
        Mnist.ToTexture(_pixels, _currentIndex, _tex);

        var def = new NetDefinition(
            Mnist.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        _net = NetBuilder.Build(def);

        // Todo: Xavier initialization
        NetUtils.Randomize(_net, _random);

        _renderer.SetTarget(_net);
    }

    private void TrainMinibatch(Network net) {
        const int batchSize = 64;
        var batch = Mnist.GetBatch(batchSize, _pixels, _labels, _random);

        var target = new float[10];
        var dCdO = new float[10];

        // Todo: have a buffer in which to store minibatch gradients for averaging
        // Needs code restructuring to make this easy to allocate

        float avgBatchCost = 0f;
        int correctLabels = 0;

        for (int i = 0; i < batch.Labels.Length; i++) {
            // Copy image to input layer
            for (int p = 0; p < Mnist.ImgDims; p++) {
                net.Input[p] = batch.Images[i][p];
            }

            NetUtils.Forward(net);

            int outputClass = NetUtils.GetMaxOutput(net);
            Mnist.LabelToVector(batch.Labels[i], target);

            if (outputClass == batch.Labels[i]) {
                correctLabels++;
            }

            // Calculate error between output layer and target
            Mnist.Subtract(target, net.Output, dCdO);
            float cost = Mnist.Cost(dCdO);
            avgBatchCost += cost;

            // Propagate error back
            // Calculate per-parameter gradient, store it

            NetUtils.Backward(net, target);

            NetUtils.StochasticParameterUpdate(net, 0.001f);
        }

        avgBatchCost /= (float)batchSize;
        Debug.Log("Batch: " + _batchesTrained + ", Cost: " + avgBatchCost + ", Correct: " + correctLabels + "/" + batchSize);

        // Update weights and biases according to averaged gradient and learning rate 

        _batchesTrained++;
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.D)) {
            Navigate(_currentIndex + 1);
        }
        if (Input.GetKeyDown(KeyCode.A)) {
            Navigate(_currentIndex - 1);
        }

        TrainMinibatch(_net);
    }

    private void Navigate(int index) {
        _currentIndex = Mathf.Clamp(index, 0, Mnist.NumImgs);
        Mnist.ToTexture(_pixels, _currentIndex, _tex);
    }

    private void OnGUI() {
        GUI.Label(new Rect(0f, 0f, 280f, 32f), "Image: " + _currentIndex);
        GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _labels[_currentIndex]);
        GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
    }

    private void OnGizmos() {

    }
}
