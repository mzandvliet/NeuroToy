using System;
using Unity.Collections;
using UnityEngine;

/* 
Todo:

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

    private void Awake() {
        _random = new System.Random(1234);

        Mnist.Load(out _pixels, out _labels);

        _tex = new Texture2D(Mnist.Rows, Mnist.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;
        
        Mnist.ToTexture(_pixels, _currentIndex, _tex);

        var def = new NetDefinition(
            Mnist.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        var net = NetBuilder.Build(def);

        // Todo: Xavier initialization

        NetUtils.Randomize(net, _random);

        TrainMinibatch(net);
    }

    private void TrainMinibatch(Network net) {
        var batch = Mnist.GetBatch(16, _pixels, _labels, _random);

        var targetVector = new float[10];
        var outputDelta = new float[10];

        for (int i = 0; i < batch.Labels.Length; i++) {
            // Copy image to input layer
            for (int p = 0; p < Mnist.ImgDims; p++) {
                net.Input[p] = batch.Images[i][p];
            }

            NetUtils.Forward(net);

            int outputClass = NetUtils.GetMaxOutput(net);
            Mnist.LabelToVector(batch.Labels[i], targetVector);

            // Calculate error between output layer and target
            Mnist.Subtract(targetVector, net.Output, outputDelta);
            float error = Mnist.SumOfSquares(outputDelta);

            Debug.Log("Target label: " + batch.Labels[i] + ", predicted: " + outputClass + ", SoSError: " + error);

            // Propagate error back

            // Calculate per-parameter gradient, store it
        }

        // Update weights and biases according to averaged gradient and learning rate 
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.D)) {
            Navigate(_currentIndex + 1);
        }
        if (Input.GetKeyDown(KeyCode.A)) {
            Navigate(_currentIndex - 1);
        }
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
}
