using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

/*
- MNIST image y-invert in DataManager, not in user code

- Stucture for a single conv layer
    - Pointer to input structure
    - Memory for kernel and activations
    - Easy creation and wiring

- MaxPool

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
        _random = new System.Random();

        DataManager.Load();

        const int inDim = 28;
        const int kSize = 3;
        const int kDepth = 16; // Fibre
        const int kStride = 1;
        const int kPadding = 0;

        int outDim = GetOutputSize(inDim, kSize, kStride, kPadding); // Todo: derive from imgdim, kSize, kStride
        if (outDim == -1) {
            Debug.LogError("Cannot perform convolution with this configuration");
            return;
        }

        var img = new NativeArray<float>(inDim * inDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var act = new NativeArray<float>(outDim * outDim * kDepth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution kernel

        var kernel = new Kernel2D(kSize, kDepth, kStride);
        NeuralMath.RandomGaussian(_random, kernel.Values, 0f, 1f);

        // Run convolution pass

        var cj = new Conv2DJob();
        cj.input = img;
        cj.output = act;
        cj.kernel = kernel;
        cj.inDim = inDim;
        cj.outDim = outDim;
        handle = cj.Schedule(handle);

        handle.Complete();

        // Create debug textures

        _imgTex = new Texture2D(inDim, inDim, TextureFormat.ARGB32, false, true);
        _imgTex.filterMode = FilterMode.Point;
        TextureUtils.ToTexture(img, _imgTex);

        _actTex = TextureUtils.CreateTexture2DArray(outDim, outDim, kDepth);
        for (int i = 0; i < kDepth; i++) {
            TextureUtils.ToTexture(act, i, _actTex[i]);
        }

        _kernelTex = TextureUtils.CreateTexture2DArray(kSize, kSize, kDepth);
        for (int i = 0; i < kDepth; i++) {
            TextureUtils.ToTexture(kernel, i, _kernelTex[i]);
        }

        kernel.Dispose();
        img.Dispose();
        act.Dispose();
        DataManager.Unload();
    }

    private static int GetOutputSize(int inputSize, int kSize, int kStride, int kPadding) {
        float result = (inputSize - kSize + kPadding * 2.0f) / (float)kStride + 1.0f;
        if (result - (int)result < float.Epsilon) {
            return (int) result;
        }
        return -1;
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

    public static void ToTexture(NativeArray<float> img, Texture2D tex) {
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

    public static void ToTexture(NativeArray<float> act, int channel, Texture2D tex) {
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

    public static void ToTexture(Kernel2D kernel, int channel, Texture2D tex) {
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
}