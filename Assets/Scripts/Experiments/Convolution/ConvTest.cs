using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*
- MNIST image y-invert in DataManager, not in user code

- Kernels with more than one pattern, so X*Y*P

Output dimensions should be a multiple of kernel channel count
Make it such that all state is easily graphable through textures on screen

- Kernel with multiple color channels, too, so X*Y*C*P

- MaxPool
- Backprop through 1 conv layer
 */

public class ConvTest : MonoBehaviour {
    private int _imgLabel;
    private Texture2D _imgTex;
    private Texture2D[] _actTex;

    private Texture2D[] _kernelTex;

    private System.Random _random;
    
    private void Awake() {
        _random = new System.Random();

        // Load some images

        DataManager.Load();

        const int inDim = 28;
        const int kSize = 3;
        const int kChannels = 16;
        const int kStride = 1;
        const int outDim = inDim - 2; // Todo: derive from imgdim, kSize, kStride

        var img = new NativeArray<float>(inDim * inDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var act = new NativeArray<float>(outDim * outDim * kChannels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution kernel

        var kernel = new Kernel2D(kSize, kChannels, kStride);
        NeuralMath.RandomGaussian(_random, kernel.Values, 0f, 1f);

        // Run convolution pass

        var cj = new Conv2DJob();
        cj.input = img;
        cj.output = act;
        cj.k = kernel;
        handle = cj.Schedule(handle);

        handle.Complete();

        // Create debug textures

        _imgTex = new Texture2D(inDim, inDim, TextureFormat.ARGB32, false, true);
        _imgTex.filterMode = FilterMode.Point;
        ToTexture(img, _imgTex);

        _actTex = CreateTexture2DArray(outDim, outDim, kChannels);
        for (int i = 0; i < kChannels; i++) {
            ToTexture(act, i, _actTex[i]);
        }

        _kernelTex = CreateTexture2DArray(kSize, kSize, kChannels);
        for (int i = 0; i < kChannels; i++) {
            ToTexture(kernel, i, _kernelTex[i]);
        }

        kernel.Dispose();
        img.Dispose();
        act.Dispose();
        DataManager.Unload();
    }

    private static Texture2D[] CreateTexture2DArray(int x, int y, int count) {
        var array = new Texture2D[count];
        for (int i = 0; i < count; i++) {
            array[i] = new Texture2D(x, y, TextureFormat.ARGB32, false, true);
            array[i].filterMode = FilterMode.Point;
        }
        return array;
    }

    private static void ToTexture(NativeArray<float> img, Texture2D tex) {
        var colors = new Color[img.Length];

        for (int y = 0; y < tex.height; y++) {
            for (int x = 0; x < tex.width; x++) {
                float pix = img[y * tex.height + x];
                // Invert y
                colors[(tex.height - 1 - y) * tex.height + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }

    private static void ToTexture(NativeArray<float> act, int channel, Texture2D tex) {
        var colors = new Color[tex.width * tex.height];

        var slice = act.Slice(tex.width * tex.height * channel, tex.width * tex.height);

        for (int y = 0; y < tex.height; y++) {
            for (int x = 0; x < tex.width; x++) {
                float pix = slice[y * tex.height + x];
                // Invert y
                colors[(tex.height - 1 - y) * tex.height + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }

    private static void ToTexture(Kernel2D kernel, int channel, Texture2D tex) {
        var colors = new Color[kernel.Size * kernel.Size];

        int start = kernel.Size * kernel.Size * channel;

        for (int y = 0; y < kernel.Size; y++) {
            for (int x = 0; x < kernel.Size; x++) {
                float pix = kernel.Values[start + y * kernel.Size + x];
                colors[y * kernel.Size + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
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
                new Rect(marginX + kSize * i, y, kSize, kSize),
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