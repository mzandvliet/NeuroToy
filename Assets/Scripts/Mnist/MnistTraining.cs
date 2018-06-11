using System;
using Unity.Collections;
using UnityEngine;

public class MnistTraining : MonoBehaviour {
    private float[] _pixels;
    private int[] _labels;
    private Texture2D _tex;

    private int _currentIndex;

    private void Awake() {
        Mnist.Load(out _pixels, out _labels);

        _tex = new Texture2D(Mnist.Rows, Mnist.Cols, TextureFormat.ARGB32, false, true); // Lol
        _tex.filterMode = FilterMode.Point;
        
        Mnist.ToTexture(_pixels, _currentIndex, _tex);

        var def = new NetDefinition(
            Mnist.ImgDims,
            new LayerDefinition(30, LayerType.Deterministic, ActivationType.Sigmoid),
            new LayerDefinition(10, LayerType.Deterministic, ActivationType.Sigmoid));
        var net = NetBuilder.Build(def);

        NetUtils.Randomize(net, new System.Random(1234));

        TrainMinibatch(net);
    }

    private void TrainMinibatch(Network net) {
        var batch = Mnist.GetBatch(16, _pixels, _labels);

        for (int i = 0; i < batch.Labels.Length; i++) {
            // Copy image to input layer (TODO: again, this is unneccessary memory duplication)
            for (int p = 0; p < Mnist.ImgDims; p++) {
                net.Input[p] = batch.Images[i][p];
            }

            NetUtils.Forward(net);

            int outputClass = NetUtils.GetMaxOutput(net);
            Debug.Log("Target label: " + batch.Labels[i] + ", predicted: " + outputClass);
        }
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
