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
    private Texture2D _actTex;

    private Texture2D[] _kernelTex;

    private System.Random _random;
    
    private void Awake() {
        _random = new System.Random();

        // Load some images

        DataManager.Load();

        const int inDim = 28;
        const int outDim = inDim - 2; // Todo: derive from imgdim, kSize, kStride
        var img = new NativeArray<float>(inDim * inDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var act = new NativeArray<float>(outDim * outDim, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 23545;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        // Create convolution kernel

        var kernel = new Kernel2D(3, 16, 1);
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
        _actTex = new Texture2D(outDim, outDim, TextureFormat.ARGB32, false, true);
        _imgTex.filterMode = FilterMode.Point;
        _actTex.filterMode = FilterMode.Point;
        ToTexture(img, _imgTex);
        ToTexture(act, _actTex);

        _kernelTex = new Texture2D[kernel.Channels];
        for (int i = 0; i < kernel.Channels; i++) {
            _kernelTex[i] = new Texture2D(kernel.Size, kernel.Size, TextureFormat.ARGB32, false, true);
            _kernelTex[i].filterMode = FilterMode.Point;
            ToTexture(kernel, i, _kernelTex[i]);
        }

        kernel.Dispose();
        img.Dispose();
        act.Dispose();
        DataManager.Unload();
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

    private static void ToTexture(Kernel2D kernel, int idx, Texture2D tex) {
        var colors = new Color[kernel.Size * kernel.Size];

        int start = kernel.Size * kernel.Size * idx;

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
        GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _imgLabel);
        GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _imgTex, ScaleMode.ScaleToFit);
        GUI.DrawTexture(new Rect(10f, 64f + 280f, 260f, 260f), _actTex, ScaleMode.ScaleToFit);

        int kTexSize = _kernelTex[0].width * 16;
        for (int i = 0; i < _kernelTex.Length; i++) {
            GUI.DrawTexture(new Rect(10f + kTexSize * i, 64f + 280f + 260f, kTexSize, kTexSize), _kernelTex[i], ScaleMode.ScaleToFit);
        }
    }
}