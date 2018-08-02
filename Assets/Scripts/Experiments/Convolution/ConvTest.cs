using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*
- MNIST image y-invert in DataManager, not in user code

- Stucture for a single conv layer
    - Easy creation and wiring

- MaxPool? (Fallen out of favor, can get by without it for now)

- Kernel with multiple color channels, too, so X*Y*C*P

- Backprop through 1 conv layer


 */

public class ConvTest : MonoBehaviour {
    private int _imgLabel;
    private Texture2D _imgTex;
    private Texture2D[] _actTex;

    private Texture2D[] _kernelTex;

    private System.Random _random;
    
    private void Awake() {
        _random = new System.Random(1234);

        DataManager.Load();

        const int inDim = 28;
        var img = new NativeArray<float>(inDim * inDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution layers

        var l = ConvLayer2D.Create(inDim, 3, 16, 1, 0);
        if (l == null) {
            return;
        }
        var l1 = l.Value;

        l = ConvLayer2D.Create(l1.OutDim, 3, 8, 1, 0);
        if (l == null) {
            return;
        }
        var l2 = l.Value;

        NeuralMath.RandomGaussian(_random, l1.Kernel, 0f, 1f);
        NeuralMath.RandomGaussian(_random, l2.Kernel, 0f, 1f);

        // Run convolution pass

        var cj = new Conv2DJob();
        cj.input = img;
        cj.layer = l1;
        handle = cj.Schedule(handle);
        cj = new Conv2DJob();
        cj.input = l1.output;
        cj.layer = l2;
        handle = cj.Schedule(handle);

        handle.Complete();

        // Create debug textures

        _imgTex = new Texture2D(inDim, inDim, TextureFormat.ARGB32, false, true);
        _imgTex.filterMode = FilterMode.Point;
        TextureUtils.ImgToTexture(img, _imgTex);

        _actTex = TextureUtils.CreateTexture2DArray(l1.OutDim, l1.OutDim, l1.Depth);
        for (int i = 0; i < l1.Depth; i++) {
            TextureUtils.ActivationToTexture(l1.output, i, _actTex[i]);
        }

        _kernelTex = TextureUtils.CreateTexture2DArray(l1.Size, l1.Size, l1.Depth);
        for (int i = 0; i < l1.Depth; i++) {
            TextureUtils.KernelToTexture(l1, i, _kernelTex[i]);
        }

        l1.Dispose();
        l2.Dispose();
        img.Dispose();
        DataManager.Unload();
    }

    private void OnGUI() {
        float marginX = 10f;
        float marginY = 10f;
        float inSize = 28f * 4f;
        float outSize = 26f * 4f;
        float kSize = _kernelTex[0].width * 16f;

        float y = 32f;

        GUI.Label(new Rect(0f, y, inSize, 32f), "Label: " + _imgLabel);
        y += 32f + marginY;

        GUI.DrawTexture(new Rect(0f, y, inSize, inSize), _imgTex, ScaleMode.ScaleToFit);
        y += inSize + marginY;

        for (int i = 0; i < _kernelTex.Length; i++) {
            GUI.DrawTexture(
                new Rect(marginX + outSize * i, y, kSize, kSize),
                _kernelTex[i],
                ScaleMode.ScaleToFit);
        }

        y += kSize + marginY;

        for (int i = 0; i < _kernelTex.Length; i++) {
            GUI.DrawTexture(
                new Rect(marginX + outSize * i, y, outSize, outSize),
                _actTex[i],
                ScaleMode.ScaleToFit);
        }
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