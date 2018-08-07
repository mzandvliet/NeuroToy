using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*

Todo:

- Sort out the channel/depth business
    - Input image is channels=1 for greyscale, channels=3 for RGB color

- Stucture for a single conv layer
    - Easy creation and wiring

- MaxPool, or AveragePool? (Fallen out of favor, can get by without it for now)

- Backprop

- Build a system that takes networks that are arbitrarily composed out of conv and FC layers
  and builds a SGD optimizer for it.

 */

public class ConvTest : MonoBehaviour {
    private IList<ConvLayer2D> _layers;
    private NativeNetworkLayer _fcLayer;

    private int _imgLabel;
    private Texture2D _imgTex;
    
    private IList<Conv2DLayerTexture> _layerTex;

    private System.Random _random;
    
    private void Awake() {
        _random = new System.Random();

        DataManager.Load();

        const int imgSize = 28;
        const int imgDepth = 1; // 3 for RGB
        var img = new NativeArray<float>(imgSize * imgSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var h = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution layers

        _layers = new List<ConvLayer2D>();

        var l1 = ConvLayer2D.Create(imgSize, imgDepth, 7, 16, 1, 0).Value;
        _layers.Add(l1);
        var l2 = ConvLayer2D.Create(l1.OutDim, l1.OutDepth, 5, 8, 1, 0).Value;
        _layers.Add(l2);
        var l3 = ConvLayer2D.Create(l2.OutDim, l2.OutDepth, 3, 4, 1, 0).Value;
        _layers.Add(l3);

        int convOutCount = l2.OutDim * l2.OutDim * l2.OutDepth;
        Debug.Log("Conv out neuron count: " + convOutCount);

        _fcLayer = new NativeNetworkLayer(10, convOutCount);

        // Parameter initialization

        for (int i = 0; i < _layers.Count; i++) {
            NeuralMath.RandomGaussian(_random, _layers[i].Kernel, 0f, 0.2f);
            NeuralMath.RandomGaussian(_random, _layers[i].Bias, 0f, 0.2f);
        }

        NeuralMath.RandomGaussian(_random, _fcLayer.Biases, 0f, 0.2f);
        NeuralMath.RandomGaussian(_random, _fcLayer.Weights, 0f, 0.2f);

        // Forward pass

        h = ScheduleForward(img, _layers, _fcLayer, h);
        h.Complete();

        int predictedLbl = NeuralMath.ArgMax(_fcLayer.Outputs);
        Debug.Log("Prediction: " + predictedLbl);

        // Create debug textures

        _imgTex = new Texture2D(imgSize, imgSize, TextureFormat.ARGB32, false, true);
        _imgTex.filterMode = FilterMode.Point;
        TextureUtils.ImgToTexture(img, _imgTex);

        _layerTex = new List<Conv2DLayerTexture>(_layers.Count);
        for (int i = 0; i < _layers.Count; i++) {
            _layerTex.Add(new Conv2DLayerTexture(_layers[i]));
        }

        // Clean up

        for (int i = 0; i < _layers.Count; i++) {
            _layers[i].Dispose();
        }
        _fcLayer.Dispose();
        img.Dispose();
        DataManager.Unload();
    }

    private static JobHandle ScheduleForward(NativeArray<float> img, IList<ConvLayer2D> _layers, NativeNetworkLayer _fcLayer, JobHandle h) {
        // Convolution layers

        var input = img;
        for (int i = 0; i < _layers.Count; i++) {
            var cj = new Conv2DJob();
            cj.input = input;
            cj.layer = _layers[i];
            h = cj.Schedule(h);

            var bj = new AdddBias2DJob();
            bj.layer = _layers[i];
            h = bj.Schedule(h);

            var rj = new NNBurst.ReluAssignJob();
            rj.Data = _layers[i].output;
            h = rj.Schedule(h);

            input = _layers[i].output;
        }

        // Fully connected layer (todo: reuse from library)

        const int numThreads = 8;

        var b = new CopyParallelJob();
        b.From = _fcLayer.Biases;
        b.To = _fcLayer.Outputs;
        h = b.Schedule(_fcLayer.Outputs.Length, _fcLayer.Outputs.Length / numThreads, h);

        var d = new DotParallelJob();
        d.Input = _layers[2].output;
        d.Weights = _fcLayer.Weights;
        d.Output = _fcLayer.Outputs;
        h = d.Schedule(_fcLayer.Outputs.Length, _fcLayer.Outputs.Length / numThreads, h);

        var s = new SigmoidAssignParallelJob();
        s.Data = _fcLayer.Outputs;
        h = s.Schedule(_fcLayer.Outputs.Length, _fcLayer.Outputs.Length / numThreads, h);

        return h;
    }

    private void OnGUI() {
        float y = 32f;

        float imgSize = 28 * GUIConfig.imgScale;

        GUI.Label(new Rect(GUIConfig.marginX, y, imgSize, GUIConfig.lblScaleY), "Label: " + _imgLabel);
        y += GUIConfig.lblScaleY + GUIConfig.marginY;

        GUI.DrawTexture(new Rect(GUIConfig.marginX, y, imgSize, imgSize), _imgTex, ScaleMode.ScaleToFit);
        y += imgSize + GUIConfig.marginY;

        for (int i = 0; i < _layerTex.Count; i++) {
            DrawConv2DLayer(_layerTex[i], ref y);
        }
    }

    private static void DrawConv2DLayer(Conv2DLayerTexture layer, ref float y) {
        float inSize = layer.Source.InDim * GUIConfig.imgScale;
        float outSize = layer.Source.InDim * GUIConfig.imgScale;
        float kSize = layer.Kernel[0].width * GUIConfig.kernScale;

        for (int i = 0; i < layer.Kernel.Length; i++) {
            GUI.DrawTexture(
                new Rect(GUIConfig.marginX + outSize * i + 20f, y, kSize, kSize),
                layer.Kernel[i],
                ScaleMode.ScaleToFit);
        }

        y += kSize + GUIConfig.marginY;

        for (int i = 0; i < layer.Activation.Length; i++) {
            GUI.DrawTexture(
                new Rect(GUIConfig.marginX + outSize * i + 20f, y, outSize, outSize),
                layer.Activation[i],
                ScaleMode.ScaleToFit);
        }

        y += outSize + GUIConfig.marginY;
    }

    private static class GUIConfig {
        public const float marginX = 10f;
        public const float marginY = 10f;
        public const float lblScaleY = 24f;
        public const float imgScale = 3f;
        public const float kernScale = 8f;
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

    public static void KernelToTexture(ConvLayer2D layer, int channel, Texture2D tex) {
        var colors = new Color[layer.Size * layer.Size];

        int start = layer.Size * layer.Size * channel;

        for (int y = 0; y < layer.Size; y++) {
            for (int x = 0; x < layer.Size; x++) {
                float pix = layer.Kernel[start + y * layer.Size + x];
                colors[y * layer.Size + x] = new Color(pix, pix, pix, 1f);
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
    public ConvLayer2D Source;
    public Texture2D[] Activation;
    public Texture2D[] Kernel;

    public Conv2DLayerTexture(ConvLayer2D layer) {
        Source = layer;

        Activation = TextureUtils.CreateTexture2DArray(layer.OutDim, layer.OutDim, layer.OutDepth);
        for (int i = 0; i < layer.OutDepth; i++) {
            TextureUtils.ActivationToTexture(layer.output, i, Activation[i]);
        }

        Kernel = TextureUtils.CreateTexture2DArray(layer.Size, layer.Size, layer.OutDepth);
        for (int i = 0; i < layer.OutDepth; i++) {
            TextureUtils.KernelToTexture(layer, i, Kernel[i]);
        }
    }
}