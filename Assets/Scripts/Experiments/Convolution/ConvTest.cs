using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*

Bug:

- Setting stride to any value other than 1 glitches out the computation.

Todo:

- Sort out the channel/depth business
    - Input image is channels=1 for greyscale, channels=3 for RGB color

- Make a gizmo for easily configuring successive layers that have compatible
parameters.

- MaxPool, or AveragePool? (Fallen out of favor, can get by without it for now)

- Backprop

- Build a system that takes networks that are arbitrarily composed out of conv and FC layers
  and builds a SGD optimizer for it.

- RGB is probably a poor sort of way to work with color
- Pixel grids are probably a poor sort of way to do vision

- Condider convolution with even-numbered kernel width?

- Replace padding with dilation

 */

public class ConvTest : MonoBehaviour {
    private IList<ConvLayer2D> _layers;
    private FCLayer _fcLayer;

    private IList<Conv2DLayerTexture> _layerTex;

    private System.Random _random;

    private NativeArray<int> _batch;
    NativeArray<float> _targetOutputs;
    NativeArray<float> _dCdO;
    NativeArray<float> _input;

    int _epochCount;
    int _batchCount;
    float _trainingLoss;
    float _rate;

    const int OutputClassCount = 10;
    const int BatchSize = 10;
    
    private void Awake() {
        _random = new System.Random(12345);

        DataManager.Load();

        const int imgSize = 28;
        const int imgDepth = 1; // 3 for RGB

        // Create convolution layers

        _layers = new List<ConvLayer2D>();
        var l1 = ConvLayer2D.Create(imgSize, imgDepth, 3, 1, 0, 16).Value;
        _layers.Add(l1);
        var l2 = ConvLayer2D.Create(l1.OutWidth, l1.NumFilters, 5, 3, 0, 8).Value;
        _layers.Add(l2);
        var l3 = ConvLayer2D.Create(l2.OutWidth, l2.NumFilters, 3, 1, 0, 4).Value;
        _layers.Add(l3);

        var last = l3;
        int convOutCount = last.OutWidth * last.OutWidth * last.NumFilters;
        Debug.Log("Conv out neuron count: " + convOutCount);

        _fcLayer = new FCLayer(10, convOutCount);

        // Parameter initialization

        for (int i = 0; i < _layers.Count; i++) {
            NeuralMath.RandomGaussian(_random, _layers[i].Kernel, 0f, 0.25f);
            NeuralMath.RandomGaussian(_random, _layers[i].Bias, 0f, 0.1f);
        }

        NeuralMath.RandomGaussian(_random, _fcLayer.Biases, 0f, 0.1f);
        NeuralMath.RandomGaussian(_random, _fcLayer.Weights, 0f, 0.1f);

        // Create debug textures

        _layerTex = new List<Conv2DLayerTexture>(_layers.Count);
        for (int i = 0; i < _layers.Count; i++) {
            _layerTex.Add(new Conv2DLayerTexture(_layers[i]));
        }

        // Create the training structure

        _batch = new NativeArray<int>(BatchSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        _targetOutputs = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _dCdO = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _input = new NativeArray<float>(DataManager.Test.ImgDims, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    private void Start() {
        TrainMinibatch();

        for (int i = 0; i < _layerTex.Count; i++) {
            _layerTex[i].Update();
        }
    }

    private void OnDestroy() {
        for (int i = 0; i < _layers.Count; i++) {
            _layers[i].Dispose();
        }
        _fcLayer.Dispose();

        _batch.Dispose();
        _targetOutputs.Dispose();
        _dCdO.Dispose();
        _input.Dispose();

        DataManager.Unload();
    }

    private void TrainMinibatch() {
        UnityEngine.Profiling.Profiler.BeginSample("TrainMiniBatch");

        float avgTrainCost = 0f;

        DataManager.GetBatch(_batch, DataManager.Train, _random);

        // var h = NeuralJobs.ZeroGradients(_gradientsAvg);
        var h = new JobHandle();

        for (int i = 0; i < _batch.Length; i++) {
            h = NeuralJobs.CopyInput(_input, DataManager.Train, _batch[i], h);
            h = ConvolutionJobs.ForwardPass(_input, _layers, h);
            h = ConvolutionJobs.ForwardPass(_layers[_layers.Count-1].output, _fcLayer, h);

            int targetLbl = (int)DataManager.Train.Labels[_batch[i]];
            h.Complete();
            NeuralMath.ClassToOneHot(targetLbl, _targetOutputs); // Todo: job

            // handle = NeuralJobs.BackwardsPass(_net, _gradients, _inputs, _targetOutputs, handle);
            // handle = NeuralJobs.AddGradients(_gradients, _gradientsAvg, handle);
            // h.Complete();

            // Todo: backwards pass logic now does this, don't redo, just check
            NeuralMath.Subtract(_targetOutputs, _fcLayer.Outputs, _dCdO);
            float cost = NeuralMath.Cost(_dCdO);
            avgTrainCost += cost;

            int predictedLbl = NeuralMath.ArgMax(_fcLayer.Outputs);
            Debug.Log("Prediction: " + predictedLbl);
        }

        // Update weights and biases according to averaged gradient and learning rate
        _rate = 3.0f / (float)BatchSize;
        // handle = NeuralJobs.UpdateParameters(_net, _gradientsAvg, _rate, handle);
        h.Complete(); // Todo: Is this one needed?

        _batchCount++;

        avgTrainCost /= (float)BatchSize;
        _trainingLoss = (float)System.Math.Round(avgTrainCost, 6);

        UnityEngine.Profiling.Profiler.EndSample();
    }

    private void OnGUI() {
        float y = 32f;

        for (int i = 0; i < _layerTex.Count; i++) {
            DrawConv2DLayer(_layerTex[i], ref y);
        }

        GUILayout.BeginVertical(GUI.skin.box);
        {
            GUILayout.Label("Epoch: " + _epochCount);
            GUILayout.Label("Batch: " + _batchCount + "/" + (DataManager.Train.Labels.Length / BatchSize));
            GUILayout.Label("Train Loss: " + _trainingLoss);
            GUILayout.Label("Rate: " + _rate);
        }
        GUILayout.EndVertical();
    }

    private static void DrawConv2DLayer(Conv2DLayerTexture layerTex, ref float y) {
        var layer = layerTex.Layer;

        float inSize = layer.InWidth * GUIConfig.imgScale;
        float outSize = 64f;
        float outPadSize = outSize + 2f;
        float kSize = 32f;
        float kPadSize = kSize + 2f;

        for (int f = 0; f < layer.NumFilters; f++) {
            for (int fs = 0; fs < layer.InDepth; fs++) {
                GUI.DrawTexture(
                    new Rect(GUIConfig.marginX + outPadSize * f, y + kPadSize * fs, kSize, kSize),
                    layerTex.KernTex[f * layer.InDepth + fs],
                    ScaleMode.ScaleToFit);
            }
        }

        y += kPadSize*layer.InDepth + GUIConfig.marginY;

        for (int i = 0; i < layerTex.ActTex.Length; i++) {
            GUI.DrawTexture(
                new Rect(GUIConfig.marginX + outPadSize * i, y, outSize, outSize),
                layerTex.ActTex[i],
                ScaleMode.ScaleToFit);
        }

        y += outSize + GUIConfig.marginY;
    }

    private static class GUIConfig {
        public const float marginX = 10f;
        public const float marginY = 10f;
        public const float lblScaleY = 24f;
        public const float imgScale = 3f;
        public const float kernScale = 32f;
    }
}

public static class TextureUtils {
    public static Texture2D[] CreateTexture2DArray(int x, int y, int count) {
        var array = new Texture2D[count];
        for (int i = 0; i < count; i++) {
            array[i] = new Texture2D(x, y, TextureFormat.ARGB32, false, true);
            array[i].filterMode = FilterMode.Point;
        }
        return array;
    }

    public static void ImgToTexture(NativeArray<float> img, Texture2D tex) {
        var colors = new Color[img.Length];

        for (int y = 0; y < tex.height; y++) {
            for (int x = 0; x < tex.width; x++) {
                float pix = img[y * tex.height + x];
                colors[y * tex.height + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }

    public static void ActivationToTexture(NativeArray<float> act, int channel, Texture2D tex) {
        var colors = new Color[tex.width * tex.height];

        var slice = act.Slice(tex.width * tex.height * channel, tex.width * tex.height);

        for (int y = 0; y < tex.height; y++) {
            for (int x = 0; x < tex.width; x++) {
                float pix = slice[y * tex.height + x];
                colors[y * tex.height + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }

    // Todo: support for the multiple filter slices
    public static void KernelToTexture(ConvLayer2D layer, int filter, int fslice, Texture2D tex) {
        var colors = new Color[layer.KWidth * layer.KWidth];

        int start = 
            layer.KWidth * layer.KWidth * layer.InDepth * filter +
            layer.KWidth * layer.KWidth * fslice;

        for (int y = 0; y < layer.KWidth; y++) {
            for (int x = 0; x < layer.KWidth; x++) {
                float pix = layer.Kernel[start + y * layer.KWidth + x];
                colors[y * layer.KWidth + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }
}


/* Todo
 - Use this, it's perfect for this case: https://docs.unity3d.com/ScriptReference/Texture2DArray.html
 */
public class Conv2DLayerTexture {
    public ConvLayer2D Layer;
    public Texture2D[] ActTex;
    public Texture2D[] KernTex;

    public Conv2DLayerTexture(ConvLayer2D layer) {
        Layer = layer;
        ActTex = TextureUtils.CreateTexture2DArray(layer.OutWidth, layer.OutWidth, layer.NumFilters);
        KernTex = TextureUtils.CreateTexture2DArray(layer.KWidth, layer.KWidth, layer.NumFilters * layer.InDepth);
    }

    public void Update() {
        for (int i = 0; i < Layer.NumFilters; i++) {
            TextureUtils.ActivationToTexture(Layer.output, i, ActTex[i]);
        }

        for (int f = 0; f < Layer.NumFilters; f++) {
            for (int fs = 0; fs < Layer.InDepth; fs++) {
                TextureUtils.KernelToTexture(Layer, f, fs, KernTex[f * Layer.InDepth + fs]);
            }
        }
    }
}