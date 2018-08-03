using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*
- Stucture for a single conv layer
    - Easy creation and wiring

- Fully connected layer for classification (or a conv layer with a kernel size equal to its input size)

- MaxPool? (Fallen out of favor, can get by without it for now)

- Kernel with multiple color channels, too, so X*Y*C*P

- Backprop through 1 conv layer

- Build a minibatch sgd optimizer that takes networks that are arbitrarily composed out of conv and FC layers


 */

public class ConvTest : MonoBehaviour {
    private IList<ConvLayer2D> _layers;

    private int _imgLabel;
    private Texture2D _imgTex;
    
    private IList<Conv2DLayerTexture> _layerTex;

    private System.Random _random;
    
    private void Awake() {
        _random = new System.Random(1234);

        DataManager.Load();

        const int imgDim = 28;
        var img = new NativeArray<float>(imgDim * imgDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution layers

        _layers = new List<ConvLayer2D>();

        var l = ConvLayer2D.Create(imgDim, 7, 16, 1, 0);
        if (l == null) {
            return;
        }
        _layers.Add(l.Value);

        l = ConvLayer2D.Create(_layers[0].OutDim, 5, 16, 1, 0);
        if (l == null) {
            return;
        }
        _layers.Add(l.Value);

        l = ConvLayer2D.Create(_layers[1].OutDim, 3, 16, 1, 0);
        if (l == null) {
            return;
        }
        _layers.Add(l.Value);
        l = ConvLayer2D.Create(_layers[2].OutDim, 3, 16, 1, 0);
        if (l == null) {
            return;
        }
        _layers.Add(l.Value);

        for (int i = 0; i < _layers.Count; i++) {
            NeuralMath.RandomGaussian(_random, _layers[i].Kernel, 0f, 1f);
        }

        // Run convolution pass

        var input = img;
        for (int i = 0; i < _layers.Count; i++) {
            var j = new Conv2DJob();
            j.input = input;
            j.layer = _layers[i];
            handle = j.Schedule(handle);

            input = _layers[i].output;
        }

        handle.Complete();

        // Create debug textures

        _imgTex = new Texture2D(imgDim, imgDim, TextureFormat.ARGB32, false, true);
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
        img.Dispose();
        DataManager.Unload();
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

        Activation = TextureUtils.CreateTexture2DArray(layer.OutDim, layer.OutDim, layer.Depth);
        for (int i = 0; i < layer.Depth; i++) {
            TextureUtils.ActivationToTexture(layer.output, i, Activation[i]);
        }

        Kernel = TextureUtils.CreateTexture2DArray(layer.Size, layer.Size, layer.Depth);
        for (int i = 0; i < layer.Depth; i++) {
            TextureUtils.KernelToTexture(layer, i, Kernel[i]);
        }
    }
}